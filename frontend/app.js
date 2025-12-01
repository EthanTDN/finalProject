// url of the backend api endpoint to send the data to
const API_URL = "/predict";

// handles the form submission and sends the patient data to the backend 
async function sendData(event) {
    event.preventDefault(); // stops the form from reloading the page

    // references to the form, submit button, and output area
    const form = event.target;
    const btn  = form.querySelector('button[type="submit"]');
    const out  = document.getElementById('result');

    // disable the button to avoid multiple submissions
    btn.disabled = true;
    out.textContent = "Sending request to backend...";

    // get the form inputs and convert them to numbers
    const data = {
        Pregnancies: Number(form.pregnancies.value),
        Glucose: Number(form.glucose.value),
        BloodPressure: Number(form.bloodPressure.value),
        SkinThickness: Number(form.skinThickness.value),
        Insulin: Number(form.insulin.value),
        BMI: Number(form.bmi.value),
        DiabetesPedigreeFunction: Number(form.dpf.value),
        Age: Number(form.age.value)
    };

    
    try {
        // log the data being sent for debugging
        console.log("Sending:", data);

        // send the data to the backend API
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        // make sure the response is ok
        if (!response.ok) {
            const text = await response.text();
            throw new Error(`HTTP ${response.status}: ${text}`);
        }

        // parse the JSON response
        const result = await response.json();
        console.log("Received:", result);

        // display the results
        const probPercent = (result.diabetic_probability * 100).toFixed(1);
        const label = result.predicted_class === 1 ? "Diabetic" : "Non-diabetic";

        // format the output in the specified area
        out.textContent =
            "Input:\n" +
            JSON.stringify(data, null, 2) +
            "\n\nPrediction:\n" +
            `  Probability diabetic: ${probPercent}%\n` +
            `  Predicted class: ${label}`;
    } catch (err) {
        console.error(err);
        out.textContent = "Error calling API:\n" + err.message;
    } finally {
        btn.disabled = false;
    }
}
