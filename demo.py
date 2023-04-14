import streamlit as st

st.set_page_config(
    page_title="ViVQA",
    page_icon="👋",
)

st.write("# ViVQA Demo! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### Vietnamese Visual Question Answering

    **Final Capstone Project - AIP491 - Spring 2023**

    **Students: Le Trung Hieu, Dao Cong Tuyen. Instructor: Nguyen Quoc Trung**

    Model architecture: 

        - Image Encoder: Swin Transformer

        - Question Encoder: PhoBERT

        - Fusion Module: Cross Attention
    
    Project Github: [ViVQA](https://github.com/daocongtuyen2x/Vietnamese_Visual_Question_Answersing.git)
"""
)