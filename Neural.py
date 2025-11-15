import streamlit as st
import graphviz

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Trip Neural Net", layout="wide")

# Smaller title to save vertical space
st.markdown("## Neural Network: Should Shobhit go on a Goa trip?")
st.write("Adjust the values and weights to see how the network changes its decision.")

# ---------- LAYOUT: LEFT (inputs) | RIGHT (diagram + result) ----------
# right a bit wider so the larger diagram has space
col_left, col_right = st.columns([3, 4])

# ================= LEFT: VALUE SLIDERS + WEIGHT INPUTS ===================
with col_left:
    st.markdown("**Inputs & weights**")

    # ---- Weather ----
    st.markdown("☀ **Weather**")
    c1, c2 = st.columns([3, 1])
    with c1:
        weather = st.slider("Weather value", 0.0, 10.0, 7.0, 0.1, key="w_val")
    with c2:
        w_weather = st.number_input("w (0–1)", 0.0, 1.0, 0.30, 0.05, key="w_wt")

    # ---- Budget ----
    st.markdown("💰 **Budget**")
    c1, c2 = st.columns([3, 1])
    with c1:
        budget = st.slider("Budget value", 0.0, 10.0, 6.0, 0.1, key="b_val")
    with c2:
        w_budget = st.number_input("w (0–1) ", 0.0, 1.0, 0.40, 0.05, key="b_wt")

    # ---- Free time ----
    st.markdown("🕒 **Free time**")
    c1, c2 = st.columns([3, 1])
    with c1:
        free_t = st.slider("Free time value", 0.0, 10.0, 8.0, 0.1, key="t_val")
    with c2:
        w_free_t = st.number_input("w (0–1)  ", 0.0, 1.0, 0.20, 0.05, key="t_wt")

    # ---- Mood ----
    st.markdown("😊 **Mood**")
    c1, c2 = st.columns([3, 1])
    with c1:
        mood = st.slider("Mood value", 0.0, 10.0, 6.5, 0.1, key="m_val")
    with c2:
        w_mood = st.number_input("w (0–1)   ", 0.0, 1.0, 0.10, 0.05, key="m_wt")

    # Threshold also on left to keep right column shorter
    threshold = st.slider("Decision threshold", 0.0, 10.0, 5.0, 0.1)

# ================= RIGHT: BIGGER GRAPH + RESULT ===================
# compute once
z = (
    weather * w_weather
    + budget * w_budget
    + free_t * w_free_t
    + mood * w_mood
)
go = z >= threshold

with col_right:
    st.markdown("**Network diagram & result**")

    # ---- BIGGER BUT STILL COMPACT Graphviz diagram ----
    dot = graphviz.Digraph()

    # Larger drawing area
    dot.attr(
        rankdir="LR",
        size="25,9",        # BIGGER DIAGRAM AREA
        nodesep="0.35",
        ranksep="0.6"
    )

    # Bigger circles + bigger text
    dot.attr(
        "node",
        shape="circle",
        fixedsize="true",
        width="1.1",       # BIG CIRcles (previously 0.6)
        fontsize="12",     # BIG TEXT
        style="filled",
        fillcolor="white"
    )

    # Thicker edges + bigger arrows
    dot.attr("edge", arrowsize="1.0", penwidth="1.5", fontsize="10")

    # Input nodes
    dot.node("W", f"Weather\n{weather:.1f}")
    dot.node("B", f"Budget\n{budget:.1f}")
    dot.node("T", f"Time\n{free_t:.1f}")
    dot.node("M", f"Mood\n{mood:.1f}")

    # Neuron + Output node
    dot.node("N", "N", fillcolor="gold", fontsize="12")
    out_label = "YES" if go else "NO"
    out_color = "palegreen" if go else "lightcoral"
    dot.node("O", out_label, fillcolor=out_color, fontsize="12")

    # Edges with larger labels
    dot.edge("W", "N", label=f"{w_weather:.2f}")
    dot.edge("B", "N", label=f"{w_budget:.2f}")
    dot.edge("T", "N", label=f"{w_free_t:.2f}")
    dot.edge("M", "N", label=f"{w_mood:.2f}")
    dot.edge("N", "O", label=f"thr={threshold:.1f}")

    # Fit only natural width, don’t shrink
    st.graphviz_chart(dot, use_container_width=False)


    # ---- compact result section ----
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Weighted sum z", f"{z:.2f}")
    with c2:
        st.metric("Threshold", f"{threshold:.2f}")

    decision_text = "Go on the trip! 🎉" if go else "Better skip this trip. 😅"
    st.write(f"**Decision:** {decision_text}")
