async function analyze() {
const text = document.getElementById("text").value;
const btn = document.getElementById("analyzeBtn");
const errorEl=document.getElementById("error");
errorEl.innerText="";
if (!text.trim()) {
    errorEl.innerText="Please enter the review first..";
    return;
}
btn.disabled = true;
btn.innerText = "Analyzing...";
try {
    const res = await fetch("/analyze", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: text }),
    });
    const data = await res.json();
    document.getElementById("summary").innerText = data.summary;
    const sentiment = document.getElementById("sentiment");
    sentiment.innerText = data.sentiment;
    sentiment.className = "sentiment " + data.sentiment.toLowerCase();
} catch (err) {
    alert("Something went wrong please wait. Thanks for your patience!!");
} finally {
    btn.disabled = false;
    btn.innerText = "Analyze";
}
}
