#!/usr/bin/env python3
"""Flask web app serving live ATP match predictions."""

from datetime import datetime

from flask import Flask, jsonify, render_template

from today import _load_tour_resources, build_rows  # noqa: E402

app = Flask(__name__)

print("Loading model and historical data...")
_atp_res = _load_tour_resources("atp")
_wta_res = _load_tour_resources("wta")
print("Ready. Visit http://localhost:5001")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def data():
    rows = build_rows(_atp_res, _wta_res)
    atp = [r for r in rows if r["tour"] == "atp"]
    wta = [r for r in rows if r["tour"] == "wta"]
    return jsonify({"updated": datetime.now().isoformat(), "atp": atp, "wta": wta})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
