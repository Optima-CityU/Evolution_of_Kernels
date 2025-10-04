import os
import json
from typing import Dict
import subprocess

def load_tasks_and_test_kernels() -> Dict[str, Dict]:

    base_path = os.path.dirname(os.path.dirname(__file__))
    tasks_path = os.path.join(base_path, "data", "tasks")
    kernels_path = os.path.join(base_path, "data", "initial_kernels")
    
    results = {
        "ai": {},
        "general": {}
    }
    
    ai_tasks_path = os.path.join(tasks_path, "ai")
    ai_kernels_path = os.path.join(kernels_path, "ai")
    results["ai"] = _process_category(ai_tasks_path, ai_kernels_path)
    
    general_tasks_path = os.path.join(tasks_path, "general")
    general_kernels_path = os.path.join(kernels_path, "general")
    results["general"] = _process_category(general_tasks_path, general_kernels_path)
    
    return results

def _process_category(tasks_path: str, kernels_path: str) -> Dict:
    category_results = {}
    
    if not os.path.exists(tasks_path):
        return {"error": f"Tasks path not found: {tasks_path}"}
    
    for task_file in os.listdir(tasks_path):
        if not task_file.endswith('.json'):
            continue
            
        task_id = os.path.splitext(task_file)[0]
        task_path = os.path.join(tasks_path, task_file)
        
        try:
            with open(task_path, 'r') as f:
                task_data = json.load(f)
        except Exception as e:
            category_results[task_id] = {
                "status": "error",
                "message": f"Failed to load task: {str(e)}"
            }
            continue
        
        kernel_file = f"{task_id}.cpp"
        kernel_path = os.path.join(kernels_path, kernel_file)
        
        if not os.path.exists(kernel_path):
            category_results[task_id] = {
                "status": "missing",
                "message": f"Kernel not found: {kernel_file}"
            }
            continue
        

        compile_result = _test_kernel_compilation(kernel_path)
        
        category_results[task_id] = {
            "task": task_data,
            "kernel_path": kernel_path,
            "compile_status": compile_result["status"],
            "compile_output": compile_result["output"]
        }
    
    return category_results

def _test_kernel_compilation(kernel_path: str) -> Dict:
    """测试C++ kernel编译和执行"""
    try:
        executable_path = os.path.splitext(kernel_path)[0]
        compile_cmd = [
            "gcc", 
            kernel_path, 
            "-o", executable_path,
            "-lstdc++"
        ]
        
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return {
                "status": "compile_error",
                "output": compile_result.stderr
            }
        
        run_result = subprocess.run(
            [f"./{executable_path}"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(executable_path)
        )
        
        if run_result.returncode != 0:
            return {
                "status": "runtime_error",
                "output": run_result.stderr
            }
            
        return {
            "status": "success",
            "output": run_result.stdout
        }
        
    except FileNotFoundError:
        return {
            "status": "error",
            "output": "gcc compiler not found"
        }
    except Exception as e:
        return {
            "status": "exception",
            "output": str(e)
        }

def task_initializing():

    task_results = load_tasks_and_test_kernels()
    

    return task_results

if __name__ == "__main__":
    main()
