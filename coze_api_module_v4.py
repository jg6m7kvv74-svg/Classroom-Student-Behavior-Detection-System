# -*- coding: utf-8 -*-
"""
课堂行为检测系统 - Coze API智能分析模块 V4 (最终版)
支持空间位置分析 + 正确解析流式响应
"""

import requests
import json

# ============ 配置区域 ============
COZE_API_KEY = "pat_hvODEEYIQt0vU5HYZpQqcgvTuokxEiv9jcrJoTkVWJv23COF5PnkTZMeg1QW83Uq"
COZE_BOT_ID = "7630048616715616294"
COZE_USER_ID = "classroom_detection_system"
COZE_API_URL = "https://api.coze.cn/v3/chat"


def call_coze_api(prompt: str, timeout: int = 90) -> str:
    """
    调用Coze API（流式响应模式）
    
    Args:
        prompt: 发送给智能体的提示词
        timeout: 超时时间（秒）
    
    Returns:
        智能体的回复文本
    """
    headers = {
        "Authorization": f"Bearer {COZE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "bot_id": COZE_BOT_ID,
        "user_id": COZE_USER_ID,
        "stream": True,
        "auto_save_history": False,
        "additional_messages": [
            {
                "role": "user",
                "content": prompt,
                "content_type": "text"
            }
        ]
    }
    
    try:
        response = requests.post(
            COZE_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=timeout,
            stream=True
        )
        
        if response.status_code != 200:
            return f"⚠️ API请求失败，状态码: {response.status_code}"
        
        full_content = ""
        reasoning_content = ""
        current_event = None
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                if line_str.startswith('event:'):
                    current_event = line_str[6:].strip()
                    
                elif line_str.startswith('data:'):
                    try:
                        data = json.loads(line_str[5:].strip())
                        
                        if current_event == 'conversation.message.delta':
                            # reasoning_content是思考过程，content是最终回复
                            rc = data.get('reasoning_content', '')
                            c = data.get('content', '')
                            
                            if rc:
                                reasoning_content += rc
                            if c:
                                full_content += c
                                
                        elif current_event == 'conversation.chat.completed':
                            break
                            
                        elif current_event == 'conversation.chat.failed':
                            error = data.get('last_error', {})
                            return f"⚠️ 智能体处理失败: {error.get('msg', '未知错误')}"
                            
                    except json.JSONDecodeError:
                        pass
        
        # 返回内容（优先返回content，如果没有则返回reasoning_content）
        result = full_content if full_content else reasoning_content
        
        if result:
            return result
        else:
            return "⚠️ 智能体未返回有效内容"
            
    except requests.exceptions.Timeout:
        return "⚠️ API请求超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        return f"⚠️ 网络请求失败: {str(e)}"
    except Exception as e:
        return f"⚠️ 发生错误: {str(e)}"


def analyze_spatial_distribution(boxes_info: list, img_width: int) -> dict:
    """
    分析检测框的空间分布
    
    Args:
        boxes_info: 检测框信息列表，每个元素为 {"class": "听课", "x_center": 320}
        img_width: 图像宽度
    
    Returns:
        空间分布统计 {"左侧区域": {"听课": 3}, "中间区域": {...}, "右侧区域": {...}}
    """
    left_boundary = img_width / 3
    right_boundary = img_width * 2 / 3
    
    spatial_stats = {
        "左侧区域": {},
        "中间区域": {},
        "右侧区域": {}
    }
    
    for box in boxes_info:
        x_center = box.get("x_center", 0)
        cls_cn = box.get("class", "未知")
        
        if x_center < left_boundary:
            region = "左侧区域"
        elif x_center > right_boundary:
            region = "右侧区域"
        else:
            region = "中间区域"
        
        spatial_stats[region][cls_cn] = spatial_stats[region].get(cls_cn, 0) + 1
    
    return spatial_stats


def call_coze_api_with_spatial(behavior_data: dict, spatial_data: dict, img_width: int = 1920, img_height: int = 1080) -> str:
    """
    调用Coze API获取课堂建议（包含空间位置分析）
    
    Args:
        behavior_data: 总体行为统计 {"听课": 15, "看手机": 3}
        spatial_data: 空间分布统计 {"左侧区域": {"听课": 3}, ...}
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        智能体生成的课堂建议文本
    """
    total = sum(behavior_data.values()) if behavior_data else 0
    
    # 构建总体数据表格
    if behavior_data:
        data_rows = []
        for behavior, count in behavior_data.items():
            percentage = (count / total * 100) if total > 0 else 0
            data_rows.append(f"| {behavior} | {count}人 | {percentage:.1f}% |")
        data_table = "\n".join(data_rows)
    else:
        data_table = "| 无检测数据 | - | - |"
    
    # 构建空间分布表格
    spatial_table_rows = ["| 区域 | 行为分布 | 专注度评估 |", "|------|---------|-----------|"]
    attention_scores = {}
    
    for region, behaviors in spatial_data.items():
        region_total = sum(behaviors.values())
        attention_count = behaviors.get("听课", 0)
        attention_rate = (attention_count / region_total * 100) if region_total > 0 else 0
        
        behavior_desc = ", ".join([f"{k}{v}人" for k, v in behaviors.items()]) if behaviors else "无数据"
        
        if attention_rate >= 70:
            attention_eval = "✅ 良好"
        elif attention_rate >= 50:
            attention_eval = "⚠️ 一般"
        else:
            attention_eval = "❌ 较差"
        
        spatial_table_rows.append(f"| {region} | {behavior_desc} | {attention_eval} ({attention_rate:.0f}%) |")
        attention_scores[region] = attention_rate
    
    spatial_table = "\n".join(spatial_table_rows)
    
    # 计算关键指标
    attention_rate = (behavior_data.get("听课", 0) / total * 100) if total > 0 else 0
    phone_rate = (behavior_data.get("看手机", 0) / total * 100) if total > 0 else 0
    sleep_rate = (behavior_data.get("睡觉", 0) / total * 100) if total > 0 else 0
    distraction_rate = (behavior_data.get("低头", 0) / total * 100) if total > 0 else 0
    
    # 找出最佳和最差区域
    if attention_scores:
        best_region = max(attention_scores, key=attention_scores.get)
        worst_region = min(attention_scores, key=attention_scores.get)
        best_score = attention_scores[best_region]
        worst_score = attention_scores[worst_region]
    else:
        best_region = worst_region = "无数据"
        best_score = worst_score = 0
    
    # 构建提示词
    prompt = f"""你是一位富有教育智慧和人文关怀的课堂观察助手，拥有10年以上的教学经验。你擅长从细微的课堂行为中洞察学生的学习状态，并给出温暖、专业且有深度的教学建议。

## 📊 本次课堂行为检测数据

### 一、总体行为统计

| 行为类别 | 检测人数 | 占比 |
|---------|---------|------|
{data_table}

**📊 关键指标**:
- 检测学生总数: {total}人
- 专注听课率: {attention_rate:.1f}%
- 看手机比例: {phone_rate:.1f}%
- 睡觉比例: {sleep_rate:.1f}%
- 低头分心比例: {distraction_rate:.1f}%

### 二、教室空间分布分析（重点！）

图像尺寸: {img_width} x {img_height} 像素

{spatial_table}

**📐 空间分布洞察**:
- 专注度最高区域: {best_region} ({best_score:.0f}%)
- 专注度最低区域: {worst_region} ({worst_score:.0f}%)

---

请你基于以上数据进行分析，请按以下格式输出：

## 🎯 一、课堂整体氛围评估
（2-3句话形象描述课堂状态，有画面感）

## 🗺️ 二、空间分布深度分析
**这是最重要的部分！** 根据各区域学生状态，给出具体的空间教学建议：
- 分析问题区域的原因（视角、光线、座位等）
- 给出老师应该走向哪个位置的建议
- 如何通过空间移动改善课堂纪律

## 🔍 三、问题行为解读
分析异常行为的深层原因

## 💡 四、干预建议
给出3-5条可立即执行的教学调整建议

## 🌟 五、正向激励
如何表扬状态好的学生，带动其他同学

---

💡 风格要求：语言温暖有同理心，建议要具体可操作，适当使用emoji，回答要丰富详实。
"""

    return call_coze_api(prompt)


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试Coze API...")
    print("=" * 60)
    
    # 简单测试
    result = call_coze_api("你好，请简单回复确认。")
    # 处理Unicode编码问题
    try:
        print(f"\n回复: {result}")
    except UnicodeEncodeError:
        # 移除emoji后再打印
        import re
        result_no_emoji = re.sub(r'[\U00010000-\U0010ffff]', '', result)
        print(f"\n回复: {result_no_emoji}")
    
    # 完整测试
    print("\n" + "=" * 60)
    print("测试课堂分析功能...")
    print("=" * 60)
    
    test_behavior = {"听课": 18, "看手机": 3, "低头": 2, "睡觉": 1}
    test_spatial = {
        "左侧区域": {"听课": 3, "看手机": 2, "睡觉": 1},
        "中间区域": {"听课": 10, "低头": 1},
        "右侧区域": {"听课": 5, "看手机": 1, "站立": 1}
    }
    
    result = call_coze_api_with_spatial(test_behavior, test_spatial)
    # 处理Unicode编码问题
    try:
        print(result)
    except UnicodeEncodeError:
        # 移除emoji后再打印
        import re
        result_no_emoji = re.sub(r'[\U00010000-\U0010ffff]', '', result)
        print(result_no_emoji)
