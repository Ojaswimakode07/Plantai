from markupsafe import Markup

disease_dic = {
    'Potato___Early_blight': """
    <b>Crop:</b> Potato<br/>
    <b>Disease:</b> Early Blight<br/><br/>
    <b>Cause:</b> Alternaria solani fungus<br/><br/>
    <b>Prevention & Cure:</b>
    <ul>
      <li>Use certified disease-free seed</li>
      <li>Crop rotation</li>
      <li>Apply fungicide regularly</li>
    </ul>
    """,

    'Potato___Late_blight': """
    <b>Crop:</b> Potato<br/>
    <b>Disease:</b> Late Blight<br/><br/>
    <b>Cause:</b> Phytophthora infestans<br/><br/>
    <b>Prevention & Cure:</b>
    <ul>
      <li>Remove infected plants immediately</li>
      <li>Use resistant varieties</li>
      <li>Ensure proper drainage</li>
    </ul>
    """,

    'Tomato___Early_blight': """
    <b>Crop:</b> Tomato<br/>
    <b>Disease:</b> Early Blight<br/><br/>
    <b>Prevention:</b>
    <ul>
      <li>Avoid wet foliage</li>
      <li>Use drip irrigation</li>
      <li>Remove infected leaves</li>
    </ul>
    """,

    'Tomato___Late_blight': """
    <b>Crop:</b> Tomato<br/>
    <b>Disease:</b> Late Blight<br/><br/>
    <b>Action:</b>
    <ul>
      <li>Destroy infected plants</li>
      <li>Use fungicides</li>
    </ul>
    """,

    'Tomato___healthy': """
    <b>Crop:</b> Tomato<br/>
    <b>Status:</b> Healthy 🌱<br/><br/>
    Keep maintaining proper care!
    """
}


def get_disease_info(prediction):
    return Markup(
        disease_dic.get(
            prediction,
            f"<b>Prediction:</b> {prediction}<br/>No additional data available."
        )
    )
