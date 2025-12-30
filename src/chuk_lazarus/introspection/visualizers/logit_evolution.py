"""
Logit lens evolution visualization.

Renders token probability evolution across layers.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..logit_lens import LogitLens, TokenEvolution


def render_logit_evolution(
    lens: LogitLens,
    tokens_to_track: list[str] | None = None,
    position: int = -1,
    output_path: str | Path | None = None,
    title: str = "Logit Lens - Token Evolution",
    width: int = 900,
    height: int = 500,
) -> str:
    """
    Render token probability evolution as an HTML chart.

    Args:
        lens: LogitLens with captured states
        tokens_to_track: List of tokens to show (None = top-5 from final layer)
        position: Sequence position to analyze
        output_path: Optional path to save HTML file
        title: Title for the visualization
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        HTML string
    """
    # Get tokens to track
    if tokens_to_track is None:
        # Use top-5 from final layer
        predictions = lens.get_layer_predictions(position=position, top_k=5)
        if predictions:
            tokens_to_track = predictions[-1].top_tokens
        else:
            tokens_to_track = []

    # Track each token
    evolutions: list[TokenEvolution] = []
    for token in tokens_to_track:
        try:
            evo = lens.track_token(token, position=position)
            evolutions.append(evo)
        except (ValueError, KeyError):
            continue

    if not evolutions:
        return "<p>No token evolutions to display</p>"

    # Prepare data for chart
    layers = evolutions[0].layers if evolutions else []
    datasets = []

    colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
    ]

    for i, evo in enumerate(evolutions):
        color = colors[i % len(colors)]
        datasets.append(
            {
                "label": evo.token,
                "data": evo.probabilities,
                "color": color,
            }
        )

    # Generate HTML with Chart.js
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{html.escape(title)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: {width + 100}px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 5px 0;
            color: #333;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 20px;
        }}
        .chart-container {{
            position: relative;
            height: {height}px;
        }}
        .insights {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .insights h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
        }}
        .insight-item {{
            margin: 5px 0;
            font-size: 13px;
        }}
        .emerge {{
            color: #4daf4a;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        <div class="subtitle">Position {position if position >= 0 else "last"}</div>

        <div class="chart-container">
            <canvas id="chart"></canvas>
        </div>

        <div class="insights" id="insights">
            <h3>Emergence Analysis</h3>
            <div id="emergence-list"></div>
        </div>
    </div>

    <script>
        const layers = {layers};
        const datasets = {
        [
            {
                "label": d["label"],
                "data": d["data"],
                "borderColor": d["color"],
                "backgroundColor": d["color"] + "20",
                "fill": False,
                "tension": 0.3,
            }
            for d in datasets
        ]
    };

        // Create chart
        const ctx = document.getElementById('chart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: layers.map(l => 'L' + l),
                datasets: datasets.map(d => ({{
                    label: d.label,
                    data: d.data,
                    borderColor: d.borderColor,
                    backgroundColor: d.backgroundColor,
                    fill: d.fill,
                    tension: d.tension,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                }}))
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        title: {{
                            display: true,
                            text: 'Probability'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Layer'
                        }}
                    }}
                }}
            }}
        }});

        // Generate emergence insights
        const emergenceList = document.getElementById('emergence-list');
        datasets.forEach((d, i) => {{
            const probs = d.data;
            let emergenceLayer = null;
            for (let j = 0; j < probs.length; j++) {{
                if (probs[j] >= 0.5) {{
                    emergenceLayer = layers[j];
                    break;
                }}
            }}

            const item = document.createElement('div');
            item.className = 'insight-item';
            if (emergenceLayer !== null) {{
                item.innerHTML = `<span style="color: ${{d.borderColor}}">"${{d.label}}"</span>: ` +
                    `reaches 50% at <span class="emerge">layer ${{emergenceLayer}}</span>`;
            }} else {{
                item.innerHTML = `<span style="color: ${{d.borderColor}}">"${{d.label}}"</span>: ` +
                    `never reaches 50%`;
            }}
            emergenceList.appendChild(item);
        }});
    </script>
</body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html_content)

    return html_content


def render_logit_table(
    lens: LogitLens,
    position: int = -1,
    top_k: int = 5,
) -> str:
    """
    Render a text table of top predictions per layer.

    Args:
        lens: LogitLens with captured states
        position: Sequence position
        top_k: Number of top tokens per layer

    Returns:
        Text table
    """
    predictions = lens.get_layer_predictions(position=position, top_k=top_k)

    if not predictions:
        return "No predictions captured"

    lines = [
        f"Logit Lens - Position {predictions[0].position}",
        "=" * 70,
        "",
    ]

    for pred in predictions:
        tokens_str = " | ".join(
            f"{t[:10]:>10}:{p:.2f}" for t, p in zip(pred.top_tokens, pred.top_probs)
        )
        lines.append(f"Layer {pred.layer_idx:2d}: {tokens_str}")

    return "\n".join(lines)
