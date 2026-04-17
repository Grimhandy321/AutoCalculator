const API = "http://localhost:8000";

async function loadBrands() {
    let res = await fetch(`${API}/data`);
    let data = await res.json();

    const brandSelect = document.getElementById("brand");
    brandSelect.innerHTML = "";

    data.brands.forEach(b => {
        let opt = document.createElement("option");
        opt.value = b;
        opt.text = b;
        brandSelect.appendChild(opt);
    });

    loadModels(brandSelect.value);
}

async function loadModels(brand) {
    let res = await fetch(`${API}/models/${brand}`);
    let data = await res.json();

    const modelSelect = document.getElementById("model");
    modelSelect.innerHTML = "";

    data.models.forEach(m => {
        let opt = document.createElement("option");
        opt.value = m;
        opt.text = m;
        modelSelect.appendChild(opt);
    });
}

async function predict() {
    const file = document.getElementById("imageInput").files[0];
    let formData = new FormData();
    formData.append("file", file);

    let res = await fetch(`${API}/predict`, { method: "POST", body: formData });
    let data = await res.json();

    document.getElementById("result").innerText =
        `${data.brand} ${data.model}`;

    if (data.confidence) {
        document.getElementById("confidence").innerText =
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    }

    document.getElementById("brand").value = data.brand;
    await loadModels(data.brand);
    document.getElementById("model").value = data.model;
}

async function estimate() {
    const brand = document.getElementById("brand").value;
    const model = document.getElementById("model").value;
    const mileage = document.getElementById("mileage").value;
    const engine = document.getElementById("engine").value;
    const year = document.getElementById("year").value;

    let res = await fetch(`${API}/price`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            brand,
            model,
            mileage,
            engine,
            year
        })
    });

    let data = await res.json();

    const price = new Intl.NumberFormat("cs-CZ").format(data.price);
    document.getElementById("priceResult").innerText = price + " Kč";
}

function drawMatrix(canvasId, labels, matrix) {
    const ctx = document.getElementById(canvasId);

    new Chart(ctx, {
        type: "matrix",
        data: {
            datasets: [{
                data: matrix.flatMap((row, i) =>
                    row.map((v, j) => ({ x: j, y: i, v }))
                ),
                backgroundColor(ctx) {
                    return `rgba(13,110,253,${ctx.raw.v / 40})`;
                }
            }]
        }
    });
}

async function loadEvaluation() {
    let res = await fetch(`${API}/evaluation`);
    let evaluation = await res.json();

    new Chart(document.getElementById("accuracyChart"), {
        type: "bar",
        data: {
            labels: ["Brand", "Model", "Condition"],
            datasets: [{
                data: [
                    evaluation.accuracy.brand,
                    evaluation.accuracy.model,
                    evaluation.accuracy.condition
                ]
            }]
        }
    });

    drawMatrix(
        "brandMatrix",
        evaluation.confusion_matrices.brand.labels,
        evaluation.confusion_matrices.brand.matrix
    );

    const table = document.getElementById("samplesTable");

    evaluation.samples.slice(0, 10).forEach(s => {
        table.innerHTML += `
            <tr>
                <td>${s.listing_id}</td>
                <td>${s.pred_brand} ${s.pred_model}</td>
                <td>${s.true_brand} ${s.true_model}</td>
            </tr>
        `;
    });
}

document.addEventListener("DOMContentLoaded", () => {
    loadBrands();
    loadEvaluation();

    document.getElementById("brand").addEventListener("change", (e) => {
        loadModels(e.target.value);
    });
});
