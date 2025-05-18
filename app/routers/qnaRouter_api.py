from fastapi import APIRouter, HTTPException
from app.models.qnaModel import QnAModel
from app.utils.qnagenerator import answerGenerator, flan_grammar_correction

router = APIRouter()

@router.post("/retriveing")
def generatingAnswer(data: QnAModel):
    try:
        input_query = data.query
        corrected_query = flan_grammar_correction(input_query)
        print("corrected query --->",corrected_query)
        if not corrected_query or not corrected_query.strip():
            raise HTTPException(status_code=400, detail="query input cannot be empty.")

        response = answerGenerator(corrected_query)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
