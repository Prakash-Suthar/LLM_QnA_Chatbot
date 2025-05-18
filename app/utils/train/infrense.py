from transformers import T5Tokenizer, T5ForConditionalGeneration

modelpath = r"E:\codified\DDReg\LLM_QnA_Chatbot\app\utils\train\t5-qa-model"
model = T5ForConditionalGeneration.from_pretrained(modelpath)
tokenizer = T5Tokenizer.from_pretrained(modelpath)
# print("model load  ---- >",model)
def get_answer(question):
    input_text = "question: " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=64)
    print("Raw output IDs:", output)
    print("Decoded:", tokenizer.decode(output[0], skip_special_tokens=True))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Try
print(get_answer("What is Glaucoma?"))

