import streamlit as st
from app.ImageSearchEngine import ImageSearchEngine
from pathlib import Path
import os
# import sys
# sys.path.append(0, '')

st.set_page_config(layout="wide")
st.title('IMAGE SEARCH APP')

config_file = 'config/creds.yaml'
search_engine = ImageSearchEngine(config_file)
user_input = st.text_input('Enter user query here...')

if user_input:
    retrieval_results = search_engine.retrieve_images(user_input)
    response_images = [f'./images/{str(s.id).zfill(4)}.jpg' for s in retrieval_results]
    response_payloads = [s.payload for s in retrieval_results]

    image_cols = st.columns(5)
    for i, col in enumerate(image_cols):
        if i < len(response_images):
            col.image(response_images[i], use_container_width=True)

    caption_cols = st.columns(5)
    for i, col in enumerate(caption_cols):
        if i < len(response_payloads):
            caption = response_payloads[i].get("caption", "")
            col.caption(caption)

    st.markdown("---")

    # st.text('Not able to show explanations because of Size constraints.')
    success, explanations = search_engine.get_explanations(user_input, retrieval_results)
    explanation_cols = st.columns(5)
    if success:
        for i, col in enumerate(explanation_cols):
            key = f'image{i + 1}'
            explanation_obj = getattr(explanations, key, None)
            if explanation_obj:
                col.markdown(f"**Explanation:** {explanation_obj.Explanation}")
    else:
        # st.error("Got error")
        st.text(explanations)
