import requests
import json
from eoh import EoH
from task_solving import task_solving

SERVER_URL = "http://192.168.0.1"

def fetch_config_from_server(server_url):
    try:

        config_response = requests.get(f"{server_url}/api/config")
        config_response.raise_for_status()
        config = config_response.json()
        
        # 获取任务列表
        tasks_response = requests.get(f"{server_url}/api/tasks")
        tasks_response.raise_for_status()
        tasks = tasks_response.json()
        
        return config, tasks
    except requests.exceptions.RequestException as e:
        print(f"从服务器获取配置失败: {str(e)}")
        return None, None

def initialize_eoh(config):
    """初始化EOH实例"""
    eoh_config = config.get("eoh", {
        "selection_num": 5,
        "max_iterations": 10,
        "num_samplers": 4
    })
    
    return EoH(
        selection_num=eoh_config["selection_num"],
        max_iterations=eoh_config["max_iterations"],
        num_samplers=eoh_config["num_samplers"]
    )

def main(SERVER_URL):
    
    config, tasks = fetch_config_from_server(SERVER_URL)
    if not config or not tasks:
        print("无法获取配置或任务，程序终止")
        return
    
    print(f"获取到 {len(tasks)} 个任务")
    
    eoh = initialize_eoh(config)
    
    results = []
    for task in tasks:
        print(f"\n正在处理任务: {task.get('task_id', '未知ID')}")
        
        result = task_solving(
            config=config,
            task_data=task,
            eoh_config=config.get("eoh")
        )
        
        results.append(result)
        print(f"任务完成，最佳得分: {result['best_solutions'][0]['score'] if result['best_solutions'] else '无解'}")
    
    with open("task_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n所有任务处理完成，结果已保存到 task_results.json")

if __name__ == "__main__":
    main()
