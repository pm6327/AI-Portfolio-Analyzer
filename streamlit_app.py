import streamlit as st


st.set_page_config(
    page_title="AI Protfolio Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "A personal project by abhaypai@vt.edu",
    },
)

input_page = st.Page(
    page="Pages/1_Set Profile.py",
    title="Input",
    default=True,
)

charts_page = st.Page(page="Pages/2_Stock Charts.py", title="Stock Charts")

risk_page = st.Page(page="Pages/3_Risk Analysis.py", title="Risk Analysis")

optimal_page = st.Page(page="Pages/4_Optimal Portfolio.py", title="Optimal Portfolio")

talk_to_p = st.Page(
    page="Pages/5_Talk To Portfolio.py",
    title="Chat with Portfolio",
    icon=":material/thumb_up:",
)

model_page = st.Page(
    page="Pages/6_Model Prediction.py",
    title="Model Prediction",
    icon=":material/insights:",
)

benchmark_page = st.Page(
    page="Pages/7_Model Benchmark.py",
    title="Model Benchmark",
    icon=":material/analytics:",
)

news_page = st.Page(
    page="Pages/8_Live Market News.py",
    title="Live Market News",
    icon=":material/article:",
)


inputs = [input_page]
outputs = [charts_page, risk_page, optimal_page, model_page, benchmark_page, news_page]
chat_page = [talk_to_p]

pg = st.navigation(
    {"User Input": inputs, "Analysis": outputs, "Chat with your Portfolio": chat_page}
)

st.sidebar.link_button("See Code Here:", "https://streamlit.io/gallery")
pg.run()
