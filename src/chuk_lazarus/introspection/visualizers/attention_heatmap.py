"""
Attention heatmap visualization.

Renders attention patterns as interactive HTML heatmaps.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attention import AttentionPattern


def render_attention_heatmap(
    pattern: AttentionPattern,
    output_path: str | Path | None = None,
    title: str = "Attention Heatmap",
    head_idx: int | None = None,
    aggregate: bool = True,
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Render attention pattern as an HTML heatmap.

    Args:
        pattern: AttentionPattern to visualize
        output_path: Optional path to save HTML file
        title: Title for the visualization
        head_idx: Specific head to show (None = aggregate)
        aggregate: If True and head_idx is None, average across heads
        width: Visualization width in pixels
        height: Visualization height in pixels

    Returns:
        HTML string
    """
    # Get attention weights
    if head_idx is not None:
        weights = pattern.get_head(head_idx)[0]  # [seq, seq]
        subtitle = f"Head {head_idx}"
    elif aggregate:
        weights = pattern.aggregate()[0]  # [seq, seq]
        subtitle = f"Mean across {pattern.num_heads} heads"
    else:
        # Show all heads - just use first for now
        weights = pattern.get_head(0)[0]
        subtitle = "Head 0"

    # Convert to list for JSON
    weights_list = weights.tolist()
    tokens = pattern.tokens or [f"[{i}]" for i in range(len(weights_list))]

    # Escape tokens for HTML/JS
    escaped_tokens = [html.escape(t.replace("'", "\\'")) for t in tokens]

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{html.escape(title)}</title>
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
        .heatmap {{
            display: grid;
            gap: 1px;
            background: #ddd;
            padding: 1px;
            overflow-x: auto;
        }}
        .row {{
            display: flex;
            gap: 1px;
        }}
        .cell {{
            min-width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        .cell:hover {{
            transform: scale(1.2);
            z-index: 1;
        }}
        .header {{
            background: #f0f0f0;
            font-weight: bold;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            height: 80px;
            min-width: 30px;
        }}
        .row-header {{
            background: #f0f0f0;
            font-weight: bold;
            min-width: 80px;
            justify-content: flex-end;
            padding-right: 5px;
        }}
        .tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 100;
        }}
        .legend {{
            margin-top: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-bar {{
            height: 20px;
            width: 200px;
            background: linear-gradient(to right, #fff, #ff4444);
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        <div class="subtitle">Layer {pattern.layer_idx} - {html.escape(subtitle)}</div>

        <div class="heatmap" id="heatmap"></div>

        <div class="legend">
            <span>Low</span>
            <div class="legend-bar"></div>
            <span>High</span>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        const tokens = {escaped_tokens};
        const weights = {weights_list};
        const container = document.getElementById('heatmap');
        const tooltip = document.getElementById('tooltip');

        // Create header row
        const headerRow = document.createElement('div');
        headerRow.className = 'row';
        headerRow.innerHTML = '<div class="cell row-header">Query \\\\ Key</div>';
        tokens.forEach(token => {{
            const cell = document.createElement('div');
            cell.className = 'cell header';
            cell.textContent = token.substring(0, 8);
            cell.title = token;
            headerRow.appendChild(cell);
        }});
        container.appendChild(headerRow);

        // Create data rows
        weights.forEach((row, i) => {{
            const rowDiv = document.createElement('div');
            rowDiv.className = 'row';

            const rowHeader = document.createElement('div');
            rowHeader.className = 'cell row-header';
            rowHeader.textContent = tokens[i].substring(0, 10);
            rowHeader.title = tokens[i];
            rowDiv.appendChild(rowHeader);

            row.forEach((weight, j) => {{
                const cell = document.createElement('div');
                cell.className = 'cell';

                // Color based on weight (white to red)
                const intensity = Math.min(255, Math.floor(weight * 255 * 2));
                cell.style.backgroundColor = `rgb(255, ${{255 - intensity}}, ${{255 - intensity}})`;

                // Show weight on hover
                cell.addEventListener('mouseenter', (e) => {{
                    tooltip.style.display = 'block';
                    tooltip.textContent = `"${{tokens[i]}}" -> "${{tokens[j]}}": ${{weight.toFixed(4)}}`;
                }});
                cell.addEventListener('mousemove', (e) => {{
                    tooltip.style.left = (e.clientX + 10) + 'px';
                    tooltip.style.top = (e.clientY + 10) + 'px';
                }});
                cell.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});

                rowDiv.appendChild(cell);
            }});

            container.appendChild(rowDiv);
        }});
    </script>
</body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html_content)

    return html_content


def render_attention_summary(
    pattern: AttentionPattern,
    position: int = -1,
    top_k: int = 5,
) -> str:
    """
    Render a text summary of attention at a specific position.

    Args:
        pattern: AttentionPattern to summarize
        position: Query position to analyze
        top_k: Number of top attended tokens to show

    Returns:
        Text summary
    """
    weights = pattern.aggregate()[0]  # [seq, seq]
    tokens = pattern.tokens or [f"[{i}]" for i in range(weights.shape[0])]

    if position < 0:
        position = weights.shape[0] + position

    # Get attention distribution for this position
    attn = weights[position].tolist()

    # Get top-k
    indexed = list(enumerate(attn))
    indexed.sort(key=lambda x: x[1], reverse=True)

    lines = [
        f"Attention focus at position {position} ('{tokens[position]}')",
        "-" * 50,
    ]

    for idx, weight in indexed[:top_k]:
        bar = "#" * int(weight * 40)
        lines.append(f"  {weight:.3f} | {bar} | '{tokens[idx]}'")

    return "\n".join(lines)
