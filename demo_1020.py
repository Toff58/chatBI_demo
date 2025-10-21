# demo.py
# ============================
# Streamlit 智能问数 Demo（支持结论生成）
# ============================

import streamlit as st
import pandas as pd
import sqlite3
import os
import requests
import json
import re
import csv
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px

# # 设置中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ============================
# 配置文件和数据库
# ============================
CSV_PATH = "7月APP使用分布_processed.csv"  # CSV 文件路径
SQLITE_PATH = "app_user_distribution.db"
TABLE_NAME = "app_data"

# ============================
# 导入 CSV 到 SQLite
# ============================
def import_csv_to_sqlite():
    if not os.path.exists(SQLITE_PATH):
        df = pd.read_csv(CSV_PATH)
        df["ppl_cnt"] = pd.to_numeric(df["ppl_cnt"], errors="coerce").fillna(0).astype(int)
        conn = sqlite3.connect(SQLITE_PATH)
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
        # st.success(f"✅ 已导入 CSV 到 SQLite 数据库：{SQLITE_PATH}")

# ============================
# 定义允许值
# ============================
GENDER_MAP = {"Female": "女", "Male": "男"}
AGE_GROUPS = ["小于20岁","20-24岁","25-29岁","30-34岁","35-39岁","40-44岁",
              "45-49岁","50-54岁","55-59岁","60-64岁","65-69岁","大于70岁"]
INCOMES = ["1.5到2万","1到1.5万","20K+","3000到5000","3000元以下","5000到7500","7500到10000"]
PROVINCES = ["安徽省","北京市","福建省","甘肃省","广东省","广西壮族自治区","贵州省","海南省","河北省","河南省",
             "黑龙江省","湖北省","湖南省","吉林省","江苏省","江西省","辽宁省","内蒙古自治区","宁夏回族自治区",
             "青海省","山东省","山西省","陕西省","上海市","四川省","天津市","西藏自治区","香港特别行政区",
             "新疆维吾尔自治区","云南省","浙江省","重庆市"]

# ============================
# DeepSeek 调用
# ============================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-99af5c749d7b44d890b9aef8d5c68f3a")

def deepseek_generate_sql(question: str) -> str:
    prompt = f"""
你是一个严格的 SQL 专家，只能基于以下表结构和可取值生成 SQL。
禁止使用未提供的字段名或字段值。
要求：
- 表名为 {TABLE_NAME}
- 字段包括：app_name, category, category_new, active_month, city_tier, income, gender, province, age, ppl_cnt
# 字段定义
- app_name: 文本，如下是其中一些：
  ['抖音短视频','快手','微信','淘宝','京东','美团','支付宝','小红书','今日头条']
- category: 文本，如 ['办公管理','出行','健康医疗','金融理财','丽人美颜','美食外卖','拍摄美化','汽车服务'
,'亲子育儿','社交沟通','生活类','手机工具','网络购物','游戏','娱乐休闲','知识类','资讯']
- city_tier: 文本，只能为：
  ['一线城市','新一线城市','二线城市','三线城市','四线城市','五线城市']
- income: 文本，只能为：
  ['3000元以下','3000到5000','5000到7500','7500到10000','1到1.5万','1.5到2万','20K+']
- gender: 文本，只能为 ['男','女']
- province: 只能为
["安徽省","北京市","福建省","甘肃省","广东省","广西壮族自治区","贵州省","海南省","河北省","河南省",
             "黑龙江省","湖北省","湖南省","吉林省","江苏省","江西省","辽宁省","内蒙古自治区","宁夏回族自治区",
             "青海省","山东省","山西省","陕西省","上海市","四川省","天津市","西藏自治区","香港特别行政区",
             "新疆维吾尔自治区","云南省","浙江省","重庆市"]
- age: 文本，只能为以下分组：
  ['小于20岁','20-24岁','25-29岁','30-34岁','35-39岁','40-44岁',
   '45-49岁','50-54岁','55-59岁','60-64岁','65-69岁','大于70岁']
- ppl_cnt: 整数，表示人数
# 生成要求
1. SQL 必须为 SQLite 兼容。
2. 仅使用以上字段与取值，不得编造或简写（如'四线'→'四线城市'）。
3. 不得使用 BETWEEN 形式的年龄比较，应基于 age 字段取值过滤。
4. “下线城市” 表示 city_tier IN ('三线城市','四线城市','五线城市')
5. 输出只包含 SQL 查询语句，不得输出解释、注释或 Markdown 标记。
6. SQL 语义需完全符合问题逻辑。
7. **如果查询结果比较多只返回 top 10 的结果（按 ppl_cnt 排序）比如年轻女性最常使用的app是哪些？**。
8. 例子：
    年轻女性最常使用的app是哪些？
    SELECT app_name, sum(ppl_cnt) as user_cnt
    FROM app_data
    WHERE gender = '女' AND age IN ('小于20岁', '20-24岁', '25-29岁')
    group by app_name
    ORDER BY user_cnt DESC
    LIMIT 10
    
    手机淘宝在上线城市的用户量是？
    SELECT 
    app_name,
    SUM(ppl_cnt) AS user_count
    FROM app_data 
    WHERE app_name = '手机淘宝'
    and city_tier in ('一线城市','新一线城市','二线城市')
    GROUP BY app_name;;问题如下：
{question}
    """
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()

    if "choices" not in data or len(data["choices"]) == 0:
        st.error("❌ DeepSeek响应异常")
        return "SELECT 'API调用失败' AS warning;"

    content = data["choices"][0]["message"]["content"]
    match = re.search(r"```sql(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        match2 = re.search(r"(SELECT[\s\S]+)", content, re.IGNORECASE)
        sql = match2.group(1).strip() if match2 else None

    if not sql:
        return "SELECT '模型未生成SQL' AS warning;"

    # 性别映射
    for k,v in GENDER_MAP.items():
        sql = sql.replace(f"'{k}'", f"'{v}'")

    # 年龄映射
    sql = map_age_condition_to_groups(sql)
    return sql

# ============================
# 年龄映射函数
# ============================
def map_age_condition_to_groups(sql: str) -> str:
    match = re.search(r"age\s*([<>=!]+)\s*(\d+)", sql)
    if match:
        operator = match.group(1)
        value = int(match.group(2))
        selected_groups = []
        for g in AGE_GROUPS:
            if g.startswith("小于"):
                min_val = 0
            elif g.startswith("大于"):
                min_val = 70
            else:
                min_val = int(g.split("-")[0])
            if operator in ("<", "<=") and min_val <= value:
                selected_groups.append(g)
            elif operator in (">", ">=") and min_val >= value:
                selected_groups.append(g)
            elif operator == "=" and min_val == value:
                selected_groups.append(g)
        if selected_groups:
            age_list_str = ", ".join(f"'{a}'" for a in selected_groups)
            sql = re.sub(r"age\s*[<>=!]+\s*\d+", f"age IN ({age_list_str})", sql)
    return sql

# ============================
# 执行 SQL
# ============================
def run_sql(sql: str):
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        st.error(f"❌ SQL执行出错: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# ============================
# 生成结论函数（泛化）
# ============================
# def generate_summary(df, question: str) -> str:
#     if df.empty:
#         return "❌ 查询结果为空，无法生成结论。"

#     # 尝试找到数量指标列（常用 ppl_cnt 或 total）
#     num_cols = [c for c in df.columns if "cnt" in c or "total" in c or df[c].dtype in [int, float]]
#     if not num_cols:
#         return f"问题：{question} 的查询结果已返回，共 {len(df)} 条记录。"

#     num_col = num_cols[0]

#     # 找出数量最大的行
#     top_row = df.iloc[df[num_col].idxmax()]

#     # 找出维度字段
#     dim_cols = [c for c in df.columns if c not in num_cols]

#     if dim_cols:
#         dim_str = ", ".join([f"{c}={top_row[c]}" for c in dim_cols])
#         return f"{question} 的结果显示：最多的是 {dim_str}，数量为 {top_row[num_col]}。"
#     else:
#         return f"{question} 的结果显示：最大数量为 {top_row[num_col]}。"


def generate_summary_by_model(question, sql, df):
    if df.empty:
        return "❌ 查询结果为空，无法生成结论。"
    
    # 更智能的数值列识别（不再只靠列名）
    num_cols = [c for c in df.columns 
                if df[c].dtype in ['int64','float64','int32','float32']
                and not all(isinstance(v, (str,)) for v in df[c].head(20))]

    if not num_cols:
        return "❌ 未能识别到可用的统计数值列，无法生成结论。"

    num_col = num_cols[0]

    # Top5摘要
    top_rows = df.sort_values(by=num_col, ascending=False).head(5)
    main_dim = [c for c in df.columns if c != num_col][0] if len(df.columns) > 1 else "结果"

    data_summary = "\n".join([f"{row[main_dim]} {row[num_col]}" 
                              for _, row in top_rows.iterrows()])

    prompt = f"""
你是一个资深的产品数据分析师，请将 SQL 查询结果转化为自然语言结论。
要求：
- 语言通俗、产品化、像 BI 报告里的结论
- 如果是单一 app，就用“该 App 在 XX 人群中使用量约 X 万”
- 如果结果是Top榜单，则列出Top N
- 数量超过 1 万要自动换算为“约 X 万”
问题：
{question}
SQL：
{sql}
查询结果摘要（Top 5）：
{data_summary}

请输出中文自然语言结论：
"""

    # 调用DeepSeek
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
              "temperature":0.2}
    )

    content = response.json().get("choices",[{}])[0].get("message",{}).get("content","")
    return content.strip()


# ============================
# 生成 Plotly top10 图
# ============================
def plot_top10_interactive(df):
    num_cols = [c for c in df.columns if "cnt" in c or "total" in c]
    if not num_cols: return None
    num_col = num_cols[0]
    dim_cols = [c for c in df.columns if c not in num_cols]
    if not dim_cols: return None
    x_col = dim_cols[0]
    
    fig = px.bar(df, x=x_col, y=num_col, text=num_col, color=num_col,
                 color_continuous_scale='Blues', labels={x_col:x_col,num_col:"人数"},
                 title="Top 10 数据分布")
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)
    return fig


#log
LOG_FILE = "ask_log.csv"
LOG_COLUMNS = ["ts", "question", "sql", "result_preview"]

def save_log(question, sql, answer):
    log_file = "query_log.csv"
    file_exists = os.path.exists(log_file)

    # 用 utf-8-sig 编码
    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 如果是新文件就写表头
        if not file_exists:
            writer.writerow(["时间", "问题", "SQL", "结论"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, sql, answer])


# ============================
# 整合问答
# ============================
def ask_ai(question: str):
    sql = deepseek_generate_sql(question)
    df = run_sql(sql)
    summary = generate_summary_by_model(question, sql ,df)
    save_log(question, sql, summary)
    fig = plot_top10_interactive(df)
    return df, summary, fig
    

# ============================
# Streamlit 界面
# ============================
st.set_page_config(page_title="智能问数 Demo", layout="centered")
st.title("🧠 智能问数 Demo")
st.write("输入你的问题，例如：'年轻女性最常使用的app是哪些？'")

# 导入 CSV 到 SQLite（第一次运行）
import_csv_to_sqlite()

question = st.text_input("请输入问题")

if st.button("查询"):
    if question.strip():
        df, summary, fig = ask_ai(question)
        st.markdown(f"**结论：** {summary}")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("请输入问题后再查询！")


