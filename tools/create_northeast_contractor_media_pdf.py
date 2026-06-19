from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = r"C:\Users\Ashish\500k_agency\prospect_briefs\warm_reply_docs\NortheastContractorMedia_Nick_3_NewEngland_Contractor_Samples_2026-06-10.pdf"

PAGE_W, PAGE_H = letter
MARGIN_X = 0.62 * inch
MARGIN_TOP = 0.45 * inch
MARGIN_BOTTOM = 0.42 * inch

NAVY = colors.HexColor("#121827")
MUTED = colors.HexColor("#4B5B73")
TEAL = colors.HexColor("#00877E")
TEAL_LIGHT = colors.HexColor("#EAF8F5")
ORANGE = colors.HexColor("#FF7A00")
ORANGE_LIGHT = colors.HexColor("#FFF6EB")
LINE = colors.HexColor("#D8E0EA")
SOFT = colors.HexColor("#F4F7FB")


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="Eyebrow",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=TEAL,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=23,
        textColor=NAVY,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="Sub",
        fontName="Helvetica",
        fontSize=10.8,
        leading=13.2,
        textColor=MUTED,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=9.4,
        leading=11.8,
        textColor=NAVY,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="Small",
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.2,
        textColor=MUTED,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="Label",
        fontName="Helvetica-Bold",
        fontSize=9.4,
        leading=11.4,
        textColor=NAVY,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        fontName="Helvetica-Bold",
        fontSize=10.7,
        leading=12.4,
        textColor=TEAL,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Email",
        fontName="Helvetica",
        fontSize=9.1,
        leading=11.7,
        textColor=NAVY,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="EmailHeader",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=12.4,
        textColor=TEAL,
        spaceAfter=7,
    )
)


def p(text, style="Body"):
    return Paragraph(text.replace("\n", "<br/>"), styles[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X, 0.33 * inch, "Built for Northeast Contractor Media / Nick")
    canvas.drawRightString(PAGE_W - MARGIN_X, 0.33 * inch, f"Page {doc.page}")
    canvas.restoreState()


def source_table(items):
    rows = [[p(item, "Small")] for item in items]
    tbl = Table(rows, colWidths=[PAGE_W - 2 * MARGIN_X])
    tbl.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    return tbl


def two_col(rows):
    data = []
    for left_label, left_body, right_label, right_body in rows:
        data.append(
            [
                p(left_label, "Label"),
                p(left_body, "Body"),
                p(right_label, "Label"),
                p(right_body, "Body"),
            ]
        )
    tbl = Table(
        data,
        colWidths=[1.08 * inch, 2.52 * inch, 1.08 * inch, 2.62 * inch],
        hAlign="LEFT",
    )
    tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return tbl


def route_box(text):
    tbl = Table([[p("CONTACT INFO / ROUTE", "Section")], [p(text, "Body")]], colWidths=[PAGE_W - 1.45 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ORANGE_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.8, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def subject_box(subject):
    tbl = Table([[p("Subject:", "Label"), p(subject, "Body")]], colWidths=[1.05 * inch, PAGE_W - 2.5 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def email_box(email_text):
    tbl = Table([[p("READY-TO-SEND EMAIL DRAFT", "EmailHeader")], [p(email_text, "Email")]], colWidths=[PAGE_W - 1.45 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ORANGE_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.8, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def verification_box(status):
    tbl = Table([[p("Verification status:", "Label"), p(status, "Small")]], colWidths=[1.45 * inch, PAGE_W - 2.8 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


targets = [
    {
        "priority": "Priority 1",
        "tag": "Best missed-call fit: urgent HVAC + plumbing demand",
        "name": "Endless Energy - Massachusetts",
        "summary": "HVAC / plumbing / electrical contractor - same-day service, emergency repair paths, large service footprint",
        "route": "<b>Website:</b> goendlessenergy.com<br/><b>Decision maker:</b> Matt Kidd, CEO<br/><b>Phone:</b> (508) 501-9990  <b>Email:</b> info@goendlessenergy.com<br/><b>Route type:</b> phone-first + official service form; public inbox is secondary.",
        "rows": [
            (
                "Location proof:",
                "Official site positions Endless Energy as a Massachusetts HVAC, plumbing, and electrical provider; BBB lists Marlborough, MA and Matthew Z. Kidd as CEO.",
                "Evidence:",
                "Official site shows same-day service, urgent hot water replacement, emergency plumbing repair, AC/heating repair, 10K+ projects, and a public call/email route.",
            ),
            (
                "Why-now signal:",
                "Their offer creates high-intent inbound calls: emergency plumbing, urgent HVAC repair, heat pumps, Mass Save assessments, and same-day service requests.",
                "Why NCM fits:",
                "Northeast Contractor Media can pitch missed-call text-back and appointment follow-up as protection for calls that already have urgency and job value.",
            ),
        ],
        "verification_status": "Manually verified public route; paid email verification not run.",
        "subject": "Missed Massachusetts service calls",
        "email": "Hi Matt,\n\nEndless Energy has same-day HVAC, plumbing, and electrical demand across Massachusetts. Your site shows urgent repair paths and 10K+ projects.\n\nThat creates one clear leak: homeowners call once, and if nobody catches them fast, they book the next contractor.\n\nNortheast Contractor Media helps contractors recover missed calls with text-back and appointment follow-up.\n\nWant to see where service calls are leaking?",
        "sources": "Endless Energy homepage/contact language; BBB Endless Energy profile; Matt Kidd CEO article; Northeast Contractor Media missed-call positioning.",
    },
    {
        "priority": "Priority 2",
        "tag": "Strongest roofing scale + quote-routing gap",
        "name": "Golden Group Roofing - Massachusetts",
        "summary": "Roofing / siding / solar contractor - four offices, online estimator, repair and replacement demand",
        "route": "<b>Website:</b> goldengrouproofing.com<br/><b>Decision maker:</b> Greta Bajrami, CEO and Co-Founder<br/><b>Phone:</b> (508) 873-1884  <b>Email:</b> official quote/contact form<br/><b>Route type:</b> phone-first + online roofing-cost calculator + service form.",
        "rows": [
            (
                "Location proof:",
                "Official contact page lists Hudson, Lexington, Westborough, and Hingham office locations plus Greater Boston service areas.",
                "Evidence:",
                "Official site offers a 2-minute roofing cost calculator, roof repairs, replacement, siding, solar, and public hours of 8AM-5PM.",
            ),
            (
                "Why-now signal:",
                "They have enough demand channels to create after-hours and overflow leakage: calculator leads, roof repair requests, siding/solar inquiries, and multiple locations.",
                "Why NCM fits:",
                "NCM can pitch missed-call recovery and appointment automation as the safety net between quote intent and booked inspection.",
            ),
        ],
        "verification_status": "Manually verified public route; paid email verification not run.",
        "subject": "Missed roofing quote calls",
        "email": "Hi Greta,\n\nGolden Group has four Massachusetts locations, a 2-minute roofing cost calculator, and repair, replacement, siding, and solar demand.\n\nThat is quote traffic. The risk is simple: if a homeowner calls and nobody answers fast, another roofer gets the inspection.\n\nNortheast Contractor Media helps contractors recover missed calls with text-back and appointment follow-up.\n\nWant to check where quote calls are leaking?",
        "sources": "Golden Group homepage; Golden Group contact page; Golden Group Greta Bajrami CEO page; Golden Group LinkedIn/company activity; NCM missed-call positioning.",
    },
    {
        "priority": "Priority 3",
        "tag": "Best storm-damage and inspection-call angle",
        "name": "SkyShield Roofing - CT / RI / Eastern MA",
        "summary": "Roofing contractor - storm damage, inspections, repairs, multi-state service area",
        "route": "<b>Website:</b> skyshieldroofing.com<br/><b>Decision maker:</b> Chad Whitcomb, co-owner<br/><b>Phone:</b> 860-726-4727  <b>Email:</b> sales@skyshieldroofpro.com<br/><b>Route type:</b> phone-first + public sales inbox + inquiry path.",
        "rows": [
            (
                "Location proof:",
                "Official site lists Lisbon, CT and service across Connecticut, Rhode Island, and Eastern Massachusetts.",
                "Evidence:",
                "Official site says 300+ property owners have chosen SkyShield; BBB describes roof repairs, inspections, replacement, and storm-damage restoration.",
            ),
            (
                "Why-now signal:",
                "Storm damage, inspections, and repair inquiries are time-sensitive. If a homeowner cannot reach them fast, the next roofer can win.",
                "Why NCM fits:",
                "NCM can lead with the first-response problem: capture roof-inspection and storm-repair calls before competitors do.",
            ),
        ],
        "verification_status": "Manually verified public route; paid email verification not run.",
        "subject": "Storm repair call leakage",
        "email": "Hi Chad,\n\nSkyShield serves CT, RI, and Eastern MA with storm-damage restoration, roof inspections, repairs, and replacement.\n\nThose calls are urgent. When a homeowner sees a leak, they usually call until someone answers.\n\nNortheast Contractor Media helps contractors recover missed calls with fast text-back and appointment follow-up.\n\nWant to see where inspection calls are leaking?",
        "sources": "SkyShield official site; BBB Sky Shield Roofing profile; Chad Whitcomb public LinkedIn profile; NCM missed-call positioning.",
    },
]


def add_cover(story):
    hero = Table(
        [
            [p("NORTHEAST CONTRACTOR MEDIA NEW ENGLAND CONTRACTOR SAMPLE - CHECKED JUNE 10, 2026", "Eyebrow")],
            [p("3 New England Contractor Prospects + Campaign-Ready Drafts", "DocTitle")],
            [
                p(
                    "Built for Nick from Northeast Contractor Media's missed-call and appointment-follow-up angle: contractors with urgent inbound demand, public routes, decision makers, and a concrete reason to contact now.",
                    "Sub",
                )
            ],
            [
                p(
                    "Public signals only. I did not contact these contractors. Each sample includes company name, website, decision maker, why-fit logic, route, subject line, and the exact first-touch email I would send.",
                    "Body",
                )
            ],
        ],
        colWidths=[PAGE_W - 2 * MARGIN_X],
    )
    hero.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 22),
                ("RIGHTPADDING", (0, 0), (-1, -1), 22),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBEFORE", (0, 3), (0, 3), 2.5, TEAL),
                ("BACKGROUND", (0, 3), (0, 3), TEAL_LIGHT),
            ]
        )
    )
    story.append(hero)
    story.append(Spacer(1, 24))
    story.append(p("How I would use these:", "Label"))
    bullets = [
        "How to use this in 10 minutes: paste the draft, send, log the call.",
        "Lead with the missed-call cost first, not generic marketing.",
        "Keep Nick's offer specific: missed-call text-back, appointment follow-up, and job-value recovery for contractors.",
        "Use the draft as the first-touch email; use the route block as the call or CRM entry.",
        "Location was checked against official New England pages, not inferred from the search query.",
    ]
    for b in bullets:
        story.append(p(b, "Body"))
    story.append(Spacer(1, 10))
    story.append(p("Source trail used:", "Label"))
    story.append(
        source_table(
            [
                "https://www.instagram.com/p/DXeaVF-FJLy/",
                "https://www.facebook.com/groups/2885633568385688/posts/4559480641000964/",
                "https://goendlessenergy.com/",
                "https://goendlessenergy.com/blog/matt-kidd-endless-energy-podcast-recap/",
                "https://www.bbb.org/us/ma/marlborough/profile/heating-and-air-conditioning/endless-energy-0021-140450",
                "https://goldengrouproofing.com/",
                "https://goldengrouproofing.com/contact",
                "https://goldengrouproofing.com/about/greta-bajrami-ceo",
                "https://www.skyshieldroofing.com/",
                "https://www.bbb.org/us/ct/jewett-city/profile/residential-roofing/sky-shield-roofing-of-new-england-llc-0111-110093003",
                "https://www.linkedin.com/in/thegreensulators",
            ]
        )
    )


def add_target(story, t):
    story.append(PageBreak())
    header = Table(
        [[p(t["priority"], "DocTitle"), p(t["tag"], "Sub")]],
        colWidths=[3.15 * inch, PAGE_W - 2 * MARGIN_X - 3.15 * inch],
    )
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("LINEBELOW", (0, 0), (-1, -1), 0.8, LINE),
            ]
        )
    )
    story.append(header)
    story.append(Spacer(1, 10))

    top = Table([[ [p(t["name"], "DocTitle"), p(t["summary"], "Sub")] ]], colWidths=[PAGE_W - 2 * MARGIN_X])
    top.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.white),
                ("LEFTPADDING", (0, 0), (0, 0), 6),
                ("RIGHTPADDING", (0, 0), (0, 0), 10),
            ]
        )
    )
    story.append(top)
    story.append(Spacer(1, 8))
    story.append(route_box(t["route"]))
    story.append(Spacer(1, 10))
    story.append(two_col(t["rows"]))
    story.append(Spacer(1, 6))
    story.append(subject_box(t["subject"]))
    story.append(Spacer(1, 10))
    story.append(email_box(t["email"]))
    story.append(Spacer(1, 6))
    story.append(verification_box(t["verification_status"]))
    story.append(Spacer(1, 6))
    src = Table([[p("Sources:", "Label"), p(t["sources"], "Small")]], colWidths=[0.9 * inch, PAGE_W - 2.25 * inch])
    src.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(src)


def main():
    doc = BaseDocTemplate(
        OUTPUT,
        pagesize=letter,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
    )
    frame = Frame(MARGIN_X, MARGIN_BOTTOM, PAGE_W - 2 * MARGIN_X, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="normal")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=footer)])
    story = []
    add_cover(story)
    for target in targets:
        add_target(story, target)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
