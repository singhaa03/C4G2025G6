document.getElementById("checkBtn").addEventListener("click", async () => {
  const text = document.getElementById("textInput").value;
  const resultDiv = document.getElementById("result");

  if (!text.trim()) {
    resultDiv.textContent = "⚠️ Please enter some text.";
    return;
  }

  resultDiv.textContent = "⏳ Checking...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();
    console.log("Full response:", data);
    console.log("Prediction value:", data.prediction);
    console.log("Confidence value:", data.confidence);
    console.log("Prediction type:", typeof data.prediction)
    resultDiv.textContent = `✔ Result: ${data.prediction} (Confidence: ${data.confidence}%)`;
  } catch (error) {
    resultDiv.textContent = "❌ Error connecting to backend.";
    console.error(error);
  }
});
