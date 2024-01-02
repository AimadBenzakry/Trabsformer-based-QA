import streamlit as st
from transformers import TFAutoModelForQuestionAnswering
import tensorflow as tf
from transformers import AutoTokenizer

# Load the pre-trained model
model = TFAutoModelForQuestionAnswering.from_pretrained("AimadBenzakry/my_qa_model")
tokenizer = AutoTokenizer.from_pretrained("AimadBenzakry/my_qa_model")

# Streamlit app
def main():
    st.title("Question Answering App")

    # User input
    context = st.text_area("Enter Context:")
    question = st.text_input("Enter Question:")

    # Predict on button click
    if st.button("Get Answer"):
        inputs = tokenizer(question, context, return_tensors="tf")
        outputs = model(**inputs)
        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)
        st.success(f"Answer: {answer}")

if __name__ == "__main__":
    main()
