import json
import os
from datetime import datetime
from typing import List, Dict

def load_result_files(result_dir: str = "data/tasks") -> List[Dict]:

    results = []
    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith("_result.json"):
                try:
                    with open(os.path.join(root, file), "r") as f:
                        results.append(json.load(f))
                except Exception as e:
                    print(f"Failed to load {file}: {str(e)}")
    return results

def generate_task_summary(result: Dict) -> Dict:

    best_solution = result.get("best_solutions", [{}])[0] if result.get("best_solutions") else {}
    
    return {
        "task_id": result.get("task_id", "unknown"),
        "status": "success" if best_solution else "failed",
        "best_score": best_solution.get("score", 0),
        "execution_time": result.get("execution_time", 0),
        "iterations": len(result.get("evolution_stats", [])),
        "error": result.get("error")
    }

def generate_overall_stats(results: List[Dict]) -> Dict:

    successful = sum(1 for r in results if r.get("status") == "success")
    total_score = sum(r.get("best_score", 0) for r in results)
    
    return {
        "total_tasks": len(results),
        "success_rate": f"{(successful / len(results)) * 100:.2f}%",
        "average_score": f"{total_score / len(results):.2f}" if results else "0",
        "start_time": min(r.get("start_time", "") for r in results if r.get("start_time")),
        "end_time": max(r.get("end_time", "") for r in results if r.get("end_time"))
    }

def gen_report(output_file: str = "final_report.json") -> Dict:

    results = load_result_files()
    if not results:
        return "error"
    
    task_summaries = [generate_task_summary(r) for r in results]
    
    overall_stats = generate_overall_stats(task_summaries)
    
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0"
        },
        "overall": overall_stats,
        "tasks": task_summaries,
        "detailed_results": results
    }
    

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report
