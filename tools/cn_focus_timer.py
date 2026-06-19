import datetime as dt
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

try:
    import winsound
except Exception:  # pragma: no cover
    winsound = None


DURATION_MINUTES = 30
TASK_TITLE = "CN4 Focus Block"
TASK_LINES = [
    "Solve 3 RSA numerical questions",
    "Solve 3 subnetting examples",
    "Proof due: photo/screenshot of completed work",
]
ABSENCE_GRACE_SECONDS = 0.6
# On Ashish's machine DirectShow lists Iriun Webcam first and Integrated Camera second.
# Use the integrated camera by default for this exam block.
CAMERA_INDEX = 1


class FocusTimer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(TASK_TITLE)
        self.root.configure(bg="#050816")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.block_close)

        self.start_time = dt.datetime.now()
        self.end_time = self.start_time + dt.timedelta(minutes=DURATION_MINUTES)
        self.done = False
        self.fullscreen = True
        self.last_face_seen = time.monotonic()
        self.last_beep = 0.0
        self.camera_ok = False
        self.latest_frame = None
        self.frame_queue: "queue.Queue[tuple[object, bool]]" = queue.Queue(maxsize=1)

        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
        ]
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.build_ui()
        self.start_camera_thread()
        self.tick()
        self.update_camera_view()

    def build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=1)

        title = tk.Label(
            self.root,
            text=TASK_TITLE,
            font=("Arial", 30, "bold"),
            fg="#f8fafc",
            bg="#050816",
        )
        title.grid(row=0, column=0, columnspan=2, pady=(28, 6))

        self.timer_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 88, "bold"),
            fg="#38bdf8",
            bg="#050816",
        )
        self.timer_label.grid(row=1, column=0, sticky="n", padx=30)

        side = tk.Frame(self.root, bg="#0f172a", padx=18, pady=18)
        side.grid(row=1, column=1, rowspan=3, sticky="nse", padx=(0, 24), pady=(20, 24))

        tk.Label(
            side,
            text="Camera watch",
            font=("Arial", 18, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        ).pack(anchor="w")

        self.camera_label = tk.Label(side, bg="#111827", width=360, height=260)
        self.camera_label.pack(pady=(12, 10))

        self.status_label = tk.Label(
            side,
            text="Starting camera...",
            font=("Arial", 12, "bold"),
            fg="#fde68a",
            bg="#0f172a",
            wraplength=340,
            justify="left",
        )
        self.status_label.pack(anchor="w", pady=(0, 14))

        btns = tk.Frame(side, bg="#0f172a")
        btns.pack(anchor="w", pady=(6, 0))

        tk.Button(
            btns,
            text="Make small",
            command=self.minimize,
            font=("Arial", 11, "bold"),
            padx=12,
            pady=8,
        ).grid(row=0, column=0, padx=(0, 8))

        tk.Button(
            btns,
            text="Fullscreen",
            command=self.toggle_fullscreen,
            font=("Arial", 11, "bold"),
            padx=12,
            pady=8,
        ).grid(row=0, column=1)

        tk.Button(
            side,
            text="DONE - proof ready",
            command=self.finish,
            font=("Arial", 13, "bold"),
            padx=16,
            pady=10,
            bg="#16a34a",
            fg="#ffffff",
        ).pack(anchor="w", pady=(18, 0))

        body = tk.Frame(self.root, bg="#050816")
        body.grid(row=2, column=0, sticky="nsew", padx=70, pady=(0, 20))

        self.time_meta = tk.Label(
            body,
            text="",
            font=("Arial", 16),
            fg="#cbd5e1",
            bg="#050816",
            justify="left",
        )
        self.time_meta.pack(anchor="w", pady=(4, 18))

        tk.Label(
            body,
            text="Today, you do not negotiate with the block. You solve, then show proof.",
            font=("Arial", 20, "bold"),
            fg="#f8fafc",
            bg="#050816",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(0, 18))

        for line in TASK_LINES:
            tk.Label(
                body,
                text="- " + line,
                font=("Arial", 18),
                fg="#e2e8f0",
                bg="#050816",
                justify="left",
            ).pack(anchor="w", pady=5)

        self.warning_label = tk.Label(
            body,
            text="",
            font=("Arial", 22, "bold"),
            fg="#fb7185",
            bg="#050816",
            wraplength=900,
            justify="left",
        )
        self.warning_label.pack(anchor="w", pady=(24, 0))

        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _event: self.minimize())

    def start_camera_thread(self) -> None:
        thread = threading.Thread(target=self.camera_loop, daemon=True)
        thread.start()

    def camera_loop(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        self.camera_ok = cap.isOpened()
        while not self.done:
            if not self.camera_ok:
                self.safe_put((None, False))
                time.sleep(1)
                continue
            ok, frame = cap.read()
            if not ok:
                self.safe_put((None, False))
                time.sleep(0.4)
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.detect_faces(gray)
            face_seen = len(faces) > 0
            if face_seen:
                self.last_face_seen = time.monotonic()
                if len(faces) > 0:
                    for (x, y, w, h) in faces[:1]:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 197, 94), 3)
            else:
                cv2.putText(frame, "NO FACE", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.safe_put((frame, face_seen))
            time.sleep(0.12)
        cap.release()

    def detect_faces(self, gray):
        found = []
        for cascade in self.face_cascades:
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(35, 35))
            if len(faces) > 0:
                found.extend(faces)
        flipped = cv2.flip(gray, 1)
        for cascade in self.face_cascades:
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(flipped, scaleFactor=1.05, minNeighbors=3, minSize=(35, 35))
            if len(faces) > 0:
                width = gray.shape[1]
                for (x, y, w, h) in faces:
                    found.append((width - x - w, y, w, h))
        return found

    def valid_eye_hits(self, eyes, width, height):
        valid = []
        for (x, y, w, h) in eyes:
            # Keep plausible human-eye hits near the upper/middle camera area.
            if y > height * 0.68:
                continue
            if x < width * 0.08 or x + w > width * 0.92:
                continue
            if w < 22 or h < 16:
                continue
            valid.append((x, y, w, h))
        return valid

    def safe_put(self, item: tuple[object, bool]) -> None:
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(item)
        except Exception:
            pass

    def update_camera_view(self) -> None:
        try:
            frame, face_seen = self.frame_queue.get_nowait()
            if frame is None:
                self.camera_label.configure(image="", text="Camera not available", fg="#f8fafc")
                self.status_label.configure(text="Camera not available. Timer still active.", fg="#fb7185")
            else:
                image = Image.fromarray(frame).resize((360, 260))
                photo = ImageTk.PhotoImage(image)
                self.latest_frame = photo
                self.camera_label.configure(image=photo)
                if face_seen:
                    self.status_label.configure(text="Face visible. Stay locked.", fg="#86efac")
                else:
                    missing = int(time.monotonic() - self.last_face_seen)
                    self.status_label.configure(text=f"No face detected for {missing}s.", fg="#fde68a")
        except queue.Empty:
            pass

        self.enforce_presence()
        if not self.done:
            self.root.after(160, self.update_camera_view)

    def tick(self) -> None:
        now = dt.datetime.now()
        remaining = self.end_time - now
        total = int(remaining.total_seconds())
        if total <= 0:
            self.timer_label.configure(text="00:00:00", fg="#fb7185")
            self.warning_label.configure(text="TIME. Stop writing, take proof photo, and report.")
            self.beep(pattern="end")
            return

        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        self.timer_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.time_meta.configure(
            text=(
                f"Start: {self.start_time:%Y-%m-%d %H:%M} Nepal Time\n"
                f"Check-in by: {self.end_time:%Y-%m-%d %H:%M} Nepal Time\n"
                f"Late after: {(self.end_time + dt.timedelta(minutes=15)):%Y-%m-%d %H:%M} Nepal Time"
            )
        )
        if total in (3600, 1800, 900, 300, 60):
            self.beep(pattern="checkpoint")
        self.root.after(1000, self.tick)

    def enforce_presence(self) -> None:
        if not self.camera_ok:
            self.warning_label.configure(text="Camera is not active. Keep the timer visible and continue. Proof is mandatory.")
            return
        missing = time.monotonic() - self.last_face_seen
        if missing >= ABSENCE_GRACE_SECONDS:
            self.warning_label.configure(text="FACE MISSING. Return to seat now.")
            if time.monotonic() - self.last_beep > 0.7:
                self.beep(pattern="absence")
                self.last_beep = time.monotonic()
        else:
            self.warning_label.configure(text="")

    def beep(self, pattern: str) -> None:
        def run() -> None:
            if winsound is None:
                self.root.bell()
                return
            if pattern == "absence":
                for freq in (1800, 1200, 2200, 1400, 2400, 1200, 2200, 1800):
                    winsound.Beep(freq, 430)
                    time.sleep(0.04)
            elif pattern == "end":
                for _ in range(5):
                    for freq in (2400, 1800, 1200):
                        winsound.Beep(freq, 420)
                        time.sleep(0.04)
            else:
                for freq in (1200, 1800, 1200):
                    winsound.Beep(freq, 260)
                    time.sleep(0.04)

        threading.Thread(target=run, daemon=True).start()

    def minimize(self) -> None:
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)
        self.root.attributes("-topmost", True)
        self.root.geometry("900x520+80+80")

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        self.root.attributes("-topmost", True)

    def block_close(self) -> None:
        self.beep(pattern="absence")
        messagebox.showwarning("Block still active", "Do not close the timer. Finish the proof first.")

    def finish(self) -> None:
        if messagebox.askyesno("Proof ready?", "Only stop if proof is ready to send. Stop timer now?"):
            self.done = True
            self.root.destroy()

    def run(self) -> None:
        self.beep(pattern="checkpoint")
        self.root.mainloop()


if __name__ == "__main__":
    FocusTimer().run()
