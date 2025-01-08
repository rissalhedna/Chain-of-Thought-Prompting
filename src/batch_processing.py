from pathlib import Path
import time
from typing import List, Dict
from openai import OpenAI
from datetime import datetime


class BatchProcessor:
    def __init__(
        self, root_dir: str = "./batch_files", output_dir: str = "./output_files"
    ):
        self.client = OpenAI()
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.method_folders = [
            "auto_cot_batches",
            "kojima_batches",
            "regular_batches",
            "tree_reasoning_batches",
        ]

    def setup_output_structure(self):
        """Create mirror output directory structure"""
        for method in self.method_folders:
            (self.output_dir / method).mkdir(parents=True, exist_ok=True)

    def process_method_batches(self, method: str) -> List[Dict]:
        """Process all batch files for a specific method"""
        method_path = self.root_dir / method
        batch_jobs = []

        # Process each batch file in the method folder
        for batch_file in method_path.glob("*.jsonl"):
            file_obj = self.client.files.create(
                file=open(batch_file, "rb"), purpose="batch"
            )

            # Create batch job
            job = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "method": method,
                    "batch_file": batch_file.name,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            batch_jobs.append(
                {"job": job, "method": method, "batch_file": batch_file.name}
            )

        return batch_jobs

    def monitor_jobs(self, batch_jobs: List[Dict], check_interval: int = 900):
        """Monitor the status of batch jobs"""
        job_statuses = {job["job"].id: job for job in batch_jobs}
        completed_jobs = set()

        while len(completed_jobs) < len(batch_jobs):
            for job_id, job_info in job_statuses.items():
                if job_id in completed_jobs:
                    continue

                status = self.client.batches.retrieve(job_id)

                if status.status == "completed":
                    print(
                        f"Job completed: {job_info['method']}/{job_info['batch_file']}"
                    )
                    self.save_job_results(status, job_info)
                    completed_jobs.add(job_id)
                elif status.status == "failed":
                    print(f"Job failed: {job_info['method']}/{job_info['batch_file']}")
                    completed_jobs.add(job_id)
                else:
                    print(f"Job {job_id} status: {status.status}")

            if len(completed_jobs) < len(batch_jobs):
                time.sleep(check_interval)

    def save_job_results(self, job_status, job_info: Dict):
        """Save completed job results to output directory"""
        output_content = self.client.files.content(job_status.output_file_id).text

        # Create output file path mirroring input structure
        output_path = self.output_dir / job_info["method"] / job_info["batch_file"]

        with open(output_path, "w") as f:
            f.write(output_content)

    def process_all_batches(self):
        """Process all batch files across all methods"""
        self.setup_output_structure()
        all_jobs = []

        for method in self.method_folders:
            method_jobs = self.process_method_batches(method)
            all_jobs.extend(method_jobs)

        self.monitor_jobs(all_jobs)
