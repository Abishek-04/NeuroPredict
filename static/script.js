document.addEventListener("DOMContentLoaded", function () {
    const symptoms = [
        "Chest Pain", "Shortness of Breath", "Irregular Heartbeat",
        "Fatigue & Weakness", "Dizziness", "Swelling (Edema)",
        "Pain in Neck/Jaw/Shoulder/Back", "Excessive Sweating", "Persistent Cough",
        "Nausea/Vomiting", "High Blood Pressure", "Chest Discomfort (Activity)",
        "Cold Hands/Feet", "Snoring/Sleep Apnea", "Anxiety/Feeling of Doom"
    ];
    
    const symptomsDiv = document.getElementById("symptoms");
    symptoms.forEach(symptom => {
        symptomsDiv.innerHTML += `<label>${symptom}: <input type="checkbox" name="${symptom}" value="1"></label><br>`;
    });

    document.getElementById("predictionForm").addEventListener("submit", function (e) {
        e.preventDefault();

        let formData = { "Age": document.getElementById("age").value };
        document.querySelectorAll("#symptoms input").forEach(input => {
            formData[input.name] = input.checked ? 1 : 0;
        });

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = 
                `Risk Classification: ${data.risk_class}, Stroke Risk: ${data.risk_percent}`;
        });
    });
});
