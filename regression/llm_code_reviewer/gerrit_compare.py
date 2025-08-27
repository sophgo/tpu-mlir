import json
import os
from typing import List, Dict, Any
import re
import sys

import requests
from openai import OpenAI
# 可选：若必须 verify=False，则关闭相关警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def llm_code_review(message,api_key):
    try:
        client = OpenAI(
            api_key=api_key, # os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://www.sophnet.com/api/open-apis/v1",
        )

        completion = client.chat.completions.create(
            model="Qwen3-Coder",
            messages=[
                {'role': 'system', 'content': '你是一个代码审查员，请review以下代码变更并给予建议。'},
                {'role': 'user', 'content': message}
                ]
        )
        # print(completion.choices[0].message.content)
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    return completion.choices[0].message.content

def strip_gerrit_json(text: str) -> str:
    # Gerrit REST 会返回前缀 ")]}'\n"
    prefix = ")]}'"
    if text.startswith(prefix):
        # 常见是前缀加换行，稳妥切一行
        return text.split("\n", 1)[1] if "\n" in text else ""
    return text

def get_info_from_project(project_name):
    # 获取 project 信息，拿当前 change 和 revision
    change_url = f"{base_url}/a/changes/?q=status:open+project:{project_name}&o=CURRENT_REVISION&n=1"
    resp = requests.get(change_url, auth=auth, verify=False)
    resp.raise_for_status()
    try:
        change_data = json.loads(strip_gerrit_json(resp.text))[0]
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Decode change response failed: {e}\nRaw: {resp.text[:200]}")
    change_id = change_data.get("_number")
    revision_id = change_data.get("current_revision")

    return change_id, revision_id

def latest_repository_compare(base_url: str, auth, change_id: str, revision_id: str) -> List[str]:
    gerrit_session = requests.Session()
    gerrit_session.auth = auth
    # 如需开启证书验证，将 verify=True
    gerrit_session.verify = False

    # 列出文件
    file_url = f"{base_url}/a/changes/{change_id}/revisions/{revision_id}/files/"
    file_resp = gerrit_session.get(file_url)
    file_resp.raise_for_status()
    file_data: Dict[str, Any] = json.loads(strip_gerrit_json(file_resp.text))

    # Gerrit 文件列表包含特殊键，比如 "/COMMIT_MSG" "/MERGE_LIST"
    special_keys = {"/COMMIT_MSG", "/MERGE_LIST"}
    file_list = [k for k in file_data.keys() if k not in special_keys]
    # print(file_list)

    def is_ignored_file(path: str) -> bool:
        # 可自行扩展忽略列表
        ext_list = {".so", ".a", ".md", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz", ".bz2"}
        _, ext = os.path.splitext(path)
        return ext.lower() in ext_list

    def repository_compare(diff_json: Dict[str, Any]) -> str:
        diff_info = diff_json.get("content", [])

        def format_compare(lines: List[str], prefix: str = "") -> str:
            return "".join(f"{prefix}{line}\n" for line in lines)

        output = ""
        for i, block in enumerate(diff_info):
            if "ab" in block:
                # 保留前后文，防止过长
                if i == 0:
                    output += format_compare(block["ab"][-4:])
                elif i == len(diff_info) - 1:
                    output += format_compare(block["ab"][0:4])
                else:
                    output += format_compare(block["ab"][0:4])
                    output += format_compare(block["ab"][-4:])
            else:
                if "a" in block and "b" in block:
                    output += format_compare(block["a"], "-")
                    output += format_compare(block["b"], "+")
                elif "a" in block:
                    output += format_compare(block["a"], "-")
                elif "b" in block:
                    output += format_compare(block["b"], "+")
        return output + "\n\n\n"

    def make_message(file_path: str) -> str:
        if is_ignored_file(file_path):
            return ""

        from urllib.parse import quote
        encoded_path = quote(file_path, safe="")
        path_url = f"{file_url}/{encoded_path}/diff/"

        path_resp = gerrit_session.get(path_url)
        path_resp.raise_for_status()
        try:
            return_data = json.loads(strip_gerrit_json(path_resp.text))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Decode diff for {file_path} failed: {e}\nRaw: {path_resp.text[:200]}")

        message = f"======== file: {file_path}\n\n" + repository_compare(return_data)
        return message

    results: List[str] = []
    for file_path in file_list:
        try:
            msg = make_message(file_path)
            if msg:
                results.append(msg)
        except requests.HTTPError as e:
            results.append(f"======== file: {file_path}\n\n[HTTP error] {e}\n\n")
        except Exception as e:
            results.append(f"======== file: {file_path}\n\n[Error] {e}\n\n")

    return results

def commit_code_review(base_url, auth, change_id, revision_id, review_result):
    review_url = f"{base_url}/a/changes/{change_id}/revisions/{revision_id}/review"
    review_post = {
                    "message": review_result,
                    "labels": {
                        "Code-Review": 0
                    }
    }
    review_req = requests.post(
        review_url,
        auth=auth,
        headers={"Content-Type": "application/json"},
        data=json.dumps(review_post),
        timeout=30,
        verify=False
    )

    text = review_req.text.lstrip()  # 去掉前导空白
    if text.startswith(")]}'"):
        text = text.split("\n", 1)[1] if "\n" in text else ""

    review_req.raise_for_status()

if __name__ == "__main__":
    # Gerrit 服务器地址 账号密码
    base_url = os.getenv("GERRIT_URL") or "https://gerrit-ai.sophgo.vip:8443/"
    user = os.getenv("LLM_BOT_USER") or "your_user"
    password = os.getenv("LLM_BOT_PASS") or "your_pass"
    api_key = os.getenv("LLM_API_KEY") or "your_api_key"
    auth = (user, password)

    # project 的名称
    project_name = sys.argv[1]
    # 获取 change_id 和 revision_id
    change_id, revision_id = get_info_from_project(project_name)

    # 根据 change_id 和 revision_id 获取文件变化
    results = latest_repository_compare(base_url, auth, change_id, revision_id)

    review_result = ""
    for block in results:
        m = re.search(r"======== file:([^\n]+)\n\n", block)
        file_path = m.group(1).strip()
        # print(block)
        review_result += "*" * 10 + file_path + "*" * 10 + '\n\n' + llm_code_review(block, api_key=api_key) + '\n\n'
    #
    commit_code_review(base_url, auth, change_id, revision_id, review_result)
