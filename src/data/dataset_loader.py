"""
Dataset loader for HumanEval and MBPP.
Handles loading, preprocessing, and batching.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import yaml


@dataclass
class CodeProblem:
    """Single code generation problem."""
    task_id: str
    prompt: str
    canonical_solution: Optional[str] = None
    test: Optional[str] = None
    entry_point: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "test": self.test,
            "entry_point": self.entry_point,
        }


class DatasetLoader:
    """Load and manage code generation datasets."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['raw_data'])
        self.dataset_name = self.config['dataset']['name']
        self.dataset_config = self.config.get("dataset", {})
        self.mbpp_prompt_style = self.dataset_config.get(
            "mbpp_prompt_style",
            "signature_stub",
        )
        self.use_mbpp_challenge_tests = self.dataset_config.get(
            "use_mbpp_challenge_tests",
            False,
        )
        self.livecodebench_version = self.dataset_config.get(
            "version_tag",
            "release_v2",
        )
        self.livecodebench_prompt_style = self.dataset_config.get(
            "livecodebench_prompt_style",
            "comment_plus_starter",
        )
        self.humaneval_prompt_style = self.dataset_config.get(
            "humaneval_prompt_style",
            "canonical",
        )

    def _build_humaneval_prompt(self, prompt: str) -> str:
        """Apply prompt-format controls for HumanEval."""
        if self.humaneval_prompt_style == "canonical":
            return prompt

        lines = prompt.splitlines()
        def_index = None
        for index, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                def_index = index
                break

        if def_index is None:
            return prompt

        preamble = lines[:def_index]
        signature = lines[def_index].rstrip()

        if self.humaneval_prompt_style == "signature_only":
            rendered = preamble + [signature, "    "]
            return "\n".join(rendered)

        if self.humaneval_prompt_style == "comment_plus_signature":
            comments = []
            in_docstring = False
            for line in lines[def_index + 1 :]:
                stripped = line.strip()
                triple_count = stripped.count('"""') + stripped.count("'''")
                if triple_count:
                    if triple_count % 2 == 1:
                        in_docstring = not in_docstring
                    stripped = stripped.replace('"""', "").replace("'''", "").strip()
                if stripped:
                    comments.append(f"# {stripped}")
                elif in_docstring:
                    comments.append("#")
            rendered = preamble + comments + [signature, "    "]
            return "\n".join(rendered)

        raise ValueError(f"Unknown HumanEval prompt style: {self.humaneval_prompt_style}")

    def _infer_mbpp_signature(self, test_list: List[str]) -> Tuple[str, List[str]]:
        """Infer a function signature from MBPP assert statements."""
        fallback_name = "solution"

        if not test_list:
            return fallback_name, []

        first_test = test_list[0].strip()
        try:
            module = ast.parse(first_test)
        except SyntaxError:
            return fallback_name, []

        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                function_name = node.func.id
                argument_names = []

                for index, _ in enumerate(node.args, start=1):
                    argument_names.append(f"arg{index}")

                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    candidate = keyword.arg
                    if candidate in argument_names:
                        candidate = f"{candidate}_arg"
                    argument_names.append(candidate)

                return function_name, argument_names

        return fallback_name, []

    def _build_mbpp_prompt(self, description: str, test_list: List[str]) -> Tuple[str, str]:
        """Convert MBPP text prompts into a valid Python function stub."""
        function_name, argument_names = self._infer_mbpp_signature(test_list)
        signature = ", ".join(argument_names)

        if self.mbpp_prompt_style == "comment_only":
            prompt_lines = [f"# Task: {description.strip()}"]
            for test in test_list[:3]:
                prompt_lines.append(f"# {test.strip()}")
            prompt_lines.append("")
            return "\n".join(prompt_lines), function_name

        prompt_lines = [
            f"def {function_name}({signature}):",
            '    """',
            f"    {description.strip()}",
        ]

        if test_list:
            prompt_lines.append("")
            prompt_lines.append("    Examples:")
            for test in test_list[:3]:
                prompt_lines.append(f"    {test.strip()}")

        prompt_lines.extend([
            '    """',
            "",
        ])

        return "\n".join(prompt_lines), function_name

    def _build_livecodebench_prompt(
        self,
        question_content: str,
        starter_code: str,
    ) -> str:
        """Format LiveCodeBench tasks as completion-style Python prompts."""
        content_lines = [line.rstrip() for line in question_content.strip().splitlines()]

        if self.livecodebench_prompt_style == "question_only":
            prompt_lines = ["# Solve the following programming problem in Python 3."]
            prompt_lines.extend(f"# {line}" if line else "#" for line in content_lines)
            prompt_lines.extend(["", ""])
            return "\n".join(prompt_lines)

        prompt_lines = ["# Solve the following programming problem in Python 3."]
        prompt_lines.extend(f"# {line}" if line else "#" for line in content_lines)
        prompt_lines.append("")

        starter_code = (starter_code or "").strip()
        if starter_code:
            prompt_lines.append(starter_code)
        else:
            prompt_lines.append("# Write your Python 3 solution below")

        prompt_lines.append("")
        return "\n".join(prompt_lines)
    
    def load_humaneval(self, num_problems: Optional[int] = None) -> List[CodeProblem]:
        """Load HumanEval dataset."""
        dataset_path = self.data_dir / "humaneval.jsonl"
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"HumanEval not found at {dataset_path}. "
                "Run: python scripts/download_data.py --dataset humaneval"
            )
        
        problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                problems.append(CodeProblem(
                    task_id=data['task_id'],
                    prompt=self._build_humaneval_prompt(data['prompt']),
                    canonical_solution=data['canonical_solution'],
                    test=data['test'],
                    entry_point=data['entry_point'],
                ))
        
        # Subset if requested
        if num_problems is not None:
            problems = problems[:num_problems]
        
        return problems
    
    def load_mbpp(self, num_problems: Optional[int] = None) -> List[CodeProblem]:
        """Load MBPP dataset."""
        dataset_path = self.data_dir / "mbpp.jsonl"
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"MBPP not found at {dataset_path}. "
                "Run: python scripts/download_data.py --dataset mbpp"
            )
        
        problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompt, entry_point = self._build_mbpp_prompt(
                    data["prompt"],
                    data["test_list"],
                )
                test_parts = []
                test_setup_code = data.get("test_setup_code", "")
                if not test_setup_code and data.get("test_imports"):
                    imports = data["test_imports"]
                    if isinstance(imports, list):
                        rendered_imports = []
                        for entry in imports:
                            entry = str(entry).strip()
                            if not entry:
                                continue
                            if entry.startswith("import ") or entry.startswith("from "):
                                rendered_imports.append(entry)
                            else:
                                rendered_imports.append(f"import {entry}")
                        test_setup_code = "\n".join(rendered_imports)
                    else:
                        test_setup_code = str(imports)
                test_setup_code = test_setup_code.strip()
                if test_setup_code:
                    test_parts.append(test_setup_code)
                test_parts.append("\n".join(data["test_list"]))
                if self.use_mbpp_challenge_tests and data.get("challenge_test_list"):
                    test_parts.append("\n".join(data["challenge_test_list"]))
                test_code = "\n\n".join(part for part in test_parts if part)

                problems.append(CodeProblem(
                    task_id=data["task_id"],
                    prompt=prompt,
                    canonical_solution=data["code"],
                    test=test_code,
                    entry_point=entry_point,
                ))
        
        # Subset if requested
        if num_problems is not None:
            problems = problems[:num_problems]

        return problems

    def load_livecodebench(self, num_problems: Optional[int] = None) -> List[CodeProblem]:
        """Load LiveCodeBench code_generation_lite."""
        dataset_path = self.data_dir / f"livecodebench_{self.livecodebench_version}.jsonl"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"LiveCodeBench not found at {dataset_path}. "
                f"Run: python scripts/download_data.py --dataset livecodebench --version_tag {self.livecodebench_version}"
            )

        problems = []
        with open(dataset_path, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt = self._build_livecodebench_prompt(
                    data.get("question_content", data.get("prompt", "")),
                    data.get("starter_code", ""),
                )
                problems.append(CodeProblem(
                    task_id=str(data["question_id"]),
                    prompt=prompt,
                    canonical_solution=data.get("canonical_solution"),
                    test=None,
                    entry_point=None,
                ))

        if num_problems is not None:
            problems = problems[:num_problems]

        return problems
    
    def load(self, dataset_name: Optional[str] = None, 
             num_problems: Optional[int] = None) -> List[CodeProblem]:
        """Load dataset based on config or parameter."""
        dataset_name = dataset_name or self.dataset_name
        
        if dataset_name == "humaneval":
            return self.load_humaneval(num_problems)
        elif dataset_name == "mbpp":
            return self.load_mbpp(num_problems)
        elif dataset_name == "livecodebench":
            return self.load_livecodebench(num_problems)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def save_problems(self, problems: List[CodeProblem], output_path: Path):
        """Save problems to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for problem in problems:
                f.write(json.dumps(problem.to_dict()) + '\n')


if __name__ == "__main__":
    # Test the loader
    loader = DatasetLoader()
    problems = loader.load_humaneval(num_problems=5)
    
    print(f"Loaded {len(problems)} problems")
    print(f"\nFirst problem:")
    print(f"Task ID: {problems[0].task_id}")
    print(f"Prompt: {problems[0].prompt[:100]}...")
