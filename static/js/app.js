async function analyze() {
    const text = document.getElementById("text").value;
    const btn = document.getElementById("analyzeBtn");
    const errorEl = document.getElementById("error");
    const summaryEl = document.getElementById("summary");
    const sentimentEl = document.getElementById("sentiment");

    errorEl.innerText = "";
    summaryEl.innerText = "";
    sentimentEl.innerText = "";
    sentimentEl.className = "sentiment";

    if (!text.trim()) {
        errorEl.innerText = "Please enter the review first.";
        return;
    }

    btn.disabled = true;
    btn.innerText = "Analyzing...";

    try {
        const res = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const data = await res.json();

        if (!res.ok) {
            errorEl.innerText = data.message || "Review is too short.";
            return;
        }

        summaryEl.innerText = data.summary;
        sentimentEl.innerText = data.sentiment;
        sentimentEl.className = "sentiment " + data.sentiment.toLowerCase();

    } catch (err) {
        errorEl.innerText = "Server error. Please try again later.";
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.innerText = "Analyze";
    }
}