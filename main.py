import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from analyzer import ImageAnalyzer

app = FastAPI()
analyzer = ImageAnalyzer()

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root() -> RedirectResponse:
    """
    Redirect to the index.html page.

    Returns:
        RedirectResponse: Redirects to the static index.html page.
    """
    return RedirectResponse(url="/static/index.html")


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...), method: str = Form(...)
) -> JSONResponse:
    """
    Analyze the uploaded image using the specified method.

    Args:
        image (UploadFile): The image file uploaded by the user.
        method (str): The method to use for analysis ("finetuned" or "standard").

    Returns:
        JSONResponse: The result of the image analysis in JSON format.
    """
    try:
        # Read image bytes directly
        image_bytes = await image.read()

        # Analyze image using the selected method
        result = analyzer.analyze(method, image_bytes)

        return JSONResponse(result)
    except Exception as e:
        # Handle any exceptions during image analysis
        return JSONResponse({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8071)
