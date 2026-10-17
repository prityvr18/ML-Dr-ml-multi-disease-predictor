from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "project_walkthrough.pdf"


def build_story():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
        alignment=1,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=18,
        firstLineIndent=-10,
        spaceAfter=6,
    )

    story = [
        Paragraph("Dr. ML - Multi-Disease Predictor", title_style),
        Paragraph("Project Walkthrough", styles["Heading1"]),
        Spacer(1, 10),
        Paragraph(
            "This project is a complete machine learning application for predicting multiple diseases using a FastAPI backend and a Streamlit frontend. It combines disease-specific training pipelines, reusable preprocessing utilities, and a user-facing clinical prediction interface.",
            body_style,
        ),
        Paragraph("1. Project Goal", heading_style),
        Paragraph(
            "The system helps healthcare users estimate the probability of two conditions: diabetes and heart disease. The application captures patient features, runs them through trained ML models, and returns a prediction with a probability score.",
            body_style,
        ),
        Paragraph("2. Core Features", heading_style),
        Paragraph("• Diabetes risk prediction using patient health metrics.", bullet_style),
        Paragraph("• Heart disease risk prediction using clinical indicators.", bullet_style),
        Paragraph("• FastAPI backend service for secure prediction requests.", bullet_style),
        Paragraph("• Streamlit frontend for interactive input and instant results.", bullet_style),
        Paragraph("• Reusable model training scripts and hyperparameter configuration.", bullet_style),
        Paragraph("• Downloadable PDF project walkthrough for developers and stakeholders.", bullet_style),
        Paragraph("3. Application Structure", heading_style),
        Paragraph(
            "The repository is organized into frontend, backend, training, and shared utility modules. This separation makes the app easier to scale, test, and maintain.",
            body_style,
        ),
        Paragraph(
            "• src/frontend: Streamlit user interfaces for diabetes and heart disease prediction.\n"
            "• src/Backend: FastAPI application, routing layer, and prediction service logic.\n"
            "• src/training: model training pipelines for each disease and configuration files.\n"
            "• src/Common: shared preprocessing utilities used during training and prediction.\n"
            "• model_dir: trained joblib model files.\n"
            "• docs: generated project documentation and exportable walkthrough PDFs.",
            body_style,
        ),
        Paragraph("4. Data Flow", heading_style),
        Paragraph(
            "The user enters clinical values in the Streamlit form. The frontend sends a JSON payload to the FastAPI endpoint /api/predict. The backend identifies the disease type, loads the corresponding trained model, and returns the prediction result with a probability value.",
            body_style,
        ),
        Paragraph("5. Training Workflow", heading_style),
        Paragraph(
            "Each disease model is trained from a CSV dataset with risk attributes and labels. The training pipeline includes preprocessing steps such as zero-to-NaN replacement, median imputation, scaling, and model fitting. Hyperparameters are loaded from YAML files to support repeatable experimentation.",
            body_style,
        ),
        Paragraph("6. Technologies Used", heading_style),
        Paragraph("• Python 3 for application logic and model training.", bullet_style),
        Paragraph("• Streamlit for interactive web-based prediction interfaces.", bullet_style),
        Paragraph("• FastAPI for backend API endpoints and request handling.", bullet_style),
        Paragraph("• Pandas, NumPy, and scikit-learn for data processing and ML workflows.", bullet_style),
        Paragraph("• Joblib to persist trained pipeline objects.", bullet_style),
        Paragraph("• ReportLab to export documents such as this walkthrough PDF.", bullet_style),
        Paragraph("7. How to Run the Project", heading_style),
        Paragraph(
            "1. Install dependencies from requirements.txt.\n"
            "2. Configure environment variables in .env using the template in env_template.txt.\n"
            "3. Start the API: uvicorn src.Backend.main:app --reload\n"
            "4. Launch the frontend: streamlit run src/frontend/app.py\n"
            "5. Open the app in the browser and submit patient details for prediction.",
            body_style,
        ),
        Paragraph("8. Prediction API", heading_style),
        Paragraph(
            "The API exposes a POST endpoint at /api/predict. It accepts the disease name and a feature dictionary, then responds with the predicted class and probability. This design keeps the model logic separate from the UI while making integration easy.",
            body_style,
        ),
        Paragraph("9. Notes for Maintainers", heading_style),
        Paragraph(
            "The project is designed for extension. New disease predictors can be added by creating a new training script, model artifact, and frontend page while reusing the same API pattern. This architecture supports future healthcare use cases with minimal structural changes.",
            body_style,
        ),
        Spacer(1, 12),
        Paragraph("Prepared for project onboarding, maintenance, and stakeholder review.", body_style),
    ]
    return story


def generate_walkthrough_pdf(output_path: Path = OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    document.build(build_story())
    print(f"PDF generated successfully at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_walkthrough_pdf()
