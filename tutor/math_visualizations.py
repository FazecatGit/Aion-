"""
Math Visualization Modules Registry.

Defines the available interactive math visualizations that the frontend can render.
Each module describes its concept, parameters,  and the data the backend can compute.
The frontend uses Three.js / Canvas to render; the backend provides evaluation endpoints.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

VISUALIZATION_MODULES = {
    # ── Geometry ──────────────────────────────────────────────────────────
    "unit_circle": {
        "name": "Unit Circle Explorer",
        "category": "Geometry",
        "icon": "📐",
        "description": "Animated unit circle with draggable angle point, live sin/cos/tan readout",
        "params": {"angle": {"type": "slider", "min": 0, "max": 360, "step": 1, "default": 45, "unit": "degrees"}},
        "outputs": ["sin", "cos", "tan", "x", "y", "radians"],
        "curriculum_topics": ["The unit circle", "Sine, cosine, tangent definitions"],
    },
    "triangle_solver": {
        "name": "Triangle Solver",
        "category": "Geometry",
        "icon": "📐",
        "description": "Input any 2 sides/angles, auto-solve the rest with color-coded sides",
        "params": {
            "side_a": {"type": "number", "default": None, "label": "Side a"},
            "side_b": {"type": "number", "default": None, "label": "Side b"},
            "side_c": {"type": "number", "default": None, "label": "Side c"},
            "angle_A": {"type": "number", "default": None, "label": "Angle A (deg)"},
            "angle_B": {"type": "number", "default": None, "label": "Angle B (deg)"},
            "angle_C": {"type": "number", "default": None, "label": "Angle C (deg)"},
        },
        "outputs": ["solved_sides", "solved_angles", "area", "perimeter"],
        "curriculum_topics": ["Law of sines", "Law of cosines", "Area of triangles using trig"],
    },
    "circle_theorems": {
        "name": "Circle Theorems Explorer",
        "category": "Geometry",
        "icon": "⭕",
        "description": "Tap a theorem, watch it animate with labeled arcs and chords",
        "params": {"theorem": {"type": "select", "options": [
            "inscribed_angle", "central_angle", "tangent_radius",
            "chord_chord", "secant_tangent", "thales"
        ]}},
        "outputs": ["animation_data", "theorem_statement", "proof_outline"],
        "curriculum_topics": ["Circle theorems", "Geometry"],
    },

    # ── Vectors & Linear Algebra ──────────────────────────────────────────
    "vector_canvas": {
        "name": "2D/3D Vector Canvas",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Add, scale, and dot-product vectors with labeled components",
        "params": {
            "vectors": {"type": "vector_list", "default": [[1, 0], [0, 1]]},
            "operation": {"type": "select", "options": ["add", "scale", "dot_product", "cross_product"]},
            "scalar": {"type": "number", "default": 1},
            "mode": {"type": "select", "options": ["2D", "3D"], "default": "2D"},
        },
        "outputs": ["result_vector", "magnitude", "angle", "visualization_data"],
        "curriculum_topics": ["Vector addition and scalar multiplication", "Dot product and cross product"],
    },
    "matrix_transform": {
        "name": "Matrix Transformation Grid",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Show how 2x2 matrices shear/rotate a grid; tied to FPS camera math",
        "params": {
            "matrix": {"type": "matrix", "rows": 2, "cols": 2, "default": [[1, 0], [0, 1]]},
            "grid_density": {"type": "slider", "min": 3, "max": 20, "default": 10},
        },
        "outputs": ["transformed_points", "determinant", "eigenvalues", "transform_type"],
        "curriculum_topics": ["Matrix as a linear transformation", "Rotation and reflection matrices"],
    },
    "eigenvalue_visualizer": {
        "name": "Eigenvalue Visualizer",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Show which directions a matrix stretches vs compresses",
        "params": {
            "matrix": {"type": "matrix", "rows": 2, "cols": 2, "default": [[2, 1], [1, 3]]},
        },
        "outputs": ["eigenvalues", "eigenvectors", "stretch_directions", "visualization_data"],
        "curriculum_topics": ["Eigenvalue equation", "Finding eigenvalues (characteristic polynomial)"],
    },
    "vector_span": {
        "name": "Vector Span & Basis",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Interactive slider to build linear combinations and see the spanned space",
        "params": {
            "v1": {"type": "vector", "default": [1, 0]},
            "v2": {"type": "vector", "default": [0, 1]},
            "c1": {"type": "slider", "min": -3, "max": 3, "step": 0.1, "default": 1},
            "c2": {"type": "slider", "min": -3, "max": 3, "step": 0.1, "default": 1},
        },
        "outputs": ["result_vector", "spans_r2", "linearly_independent"],
        "curriculum_topics": ["Linear combinations and span", "Linear independence"],
    },

    # ── Calculus ──────────────────────────────────────────────────────────
    "derivative_animator": {
        "name": "Derivative Animator",
        "category": "Calculus",
        "icon": "∫",
        "description": "Sweep a tangent line along a curve with the slope value updating live",
        "params": {
            "expression": {"type": "text", "default": "x^2", "label": "f(x)"},
            "x_point": {"type": "slider", "min": -5, "max": 5, "step": 0.1, "default": 1},
            "x_range": {"type": "range", "default": [-5, 5]},
        },
        "outputs": ["curve_points", "tangent_line", "slope", "derivative_expression"],
        "curriculum_topics": ["Definition of the derivative", "Derivatives of trig functions"],
    },
    "riemann_sum": {
        "name": "Riemann Sum Builder",
        "category": "Calculus",
        "icon": "∫",
        "description": "Slider controls rectangle count, area estimate converges to integral",
        "params": {
            "expression": {"type": "text", "default": "x^2", "label": "f(x)"},
            "n_rectangles": {"type": "slider", "min": 1, "max": 200, "step": 1, "default": 10},
            "a": {"type": "number", "default": 0, "label": "Lower bound"},
            "b": {"type": "number", "default": 2, "label": "Upper bound"},
            "method": {"type": "select", "options": ["left", "right", "midpoint"], "default": "left"},
        },
        "outputs": ["rectangles", "area_estimate", "curve_points", "exact_integral"],
        "curriculum_topics": ["Definite integrals and area", "Fundamental theorem of calculus"],
    },
    "taylor_series": {
        "name": "Taylor Series Expansion",
        "category": "Calculus",
        "icon": "∫",
        "description": "Watch polynomials approximate sin/cos step by step",
        "params": {
            "function": {"type": "select", "options": ["sin", "cos", "exp", "log(1+x)", "1/(1-x)"], "default": "sin"},
            "n_terms": {"type": "slider", "min": 1, "max": 15, "step": 1, "default": 3},
            "center": {"type": "number", "default": 0, "label": "Expansion center a"},
            "x_range": {"type": "range", "default": [-6, 6]},
        },
        "outputs": ["original_curve", "approximation_curve", "polynomial_expression", "error_curve"],
        "curriculum_topics": ["Taylor and Maclaurin series", "Power series"],
    },
    "fourier_series": {
        "name": "Fourier Series",
        "category": "Calculus",
        "icon": "∫",
        "description": "Show how sine waves stack to form complex waveforms",
        "params": {
            "waveform": {"type": "select", "options": ["square", "sawtooth", "triangle", "custom"], "default": "square"},
            "n_harmonics": {"type": "slider", "min": 1, "max": 50, "step": 1, "default": 5},
        },
        "outputs": ["component_waves", "combined_wave", "coefficients"],
        "curriculum_topics": ["Fourier Series", "Sequences and convergence"],
    },
    "parametric_curves": {
        "name": "Parametric Curves",
        "category": "Calculus",
        "icon": "∫",
        "description": "Visualize curves defined by x(t) and y(t) as t sweeps",
        "params": {
            "x_expr": {"type": "text", "default": "cos(t)", "label": "x(t)"},
            "y_expr": {"type": "text", "default": "sin(t)", "label": "y(t)"},
            "t_range": {"type": "range", "default": [0, 6.28]},
            "t_current": {"type": "slider", "min": 0, "max": 6.28, "step": 0.05, "default": 3.14},
        },
        "outputs": ["curve_points", "current_point", "velocity_vector"],
        "curriculum_topics": ["Parametric curves"],
    },

    # ── Statistics ────────────────────────────────────────────────────────
    "bell_curve": {
        "name": "Normal Distribution (Bell Curve)",
        "category": "Statistics",
        "icon": "🎲",
        "description": "Adjustable mean and standard deviation sliders, see the curve reshape live",
        "params": {
            "mu": {"type": "slider", "min": -5, "max": 5, "step": 0.1, "default": 0, "label": "Mean (mu)"},
            "sigma": {"type": "slider", "min": 0.1, "max": 3, "step": 0.1, "default": 1, "label": "Std Dev (sigma)"},
            "show_area": {"type": "toggle", "default": True, "label": "Show area under curve"},
        },
        "outputs": ["curve_points", "area_data", "percentiles"],
        "curriculum_topics": ["Normal distribution", "Standard deviation and variance"],
    },
    "regression_line": {
        "name": "Correlation & Regression",
        "category": "Statistics",
        "icon": "🎲",
        "description": "Live scatter plot with a draggable regression line",
        "params": {
            "points": {"type": "point_list", "default": [[1, 2], [2, 4], [3, 5], [4, 4], [5, 5]]},
            "show_residuals": {"type": "toggle", "default": True},
        },
        "outputs": ["regression_coefficients", "r_squared", "residuals", "best_fit_line"],
        "curriculum_topics": ["Linear regression", "Correlation and scatter plots"],
    },
    "monte_carlo_pi": {
        "name": "Monte Carlo Pi Estimator",
        "category": "Statistics",
        "icon": "🎲",
        "description": "Random points, watch pi converge visually",
        "params": {
            "n_points": {"type": "slider", "min": 10, "max": 10000, "step": 10, "default": 1000},
            "speed": {"type": "slider", "min": 1, "max": 100, "step": 1, "default": 10},
        },
        "outputs": ["points_inside", "points_outside", "pi_estimate", "convergence_history"],
        "curriculum_topics": ["Monte Carlo simulation", "Basic probability rules"],
    },

    # ── 3D & Game Math ────────────────────────────────────────────────────
    "surface_3d": {
        "name": "3D Surface Plot",
        "category": "3D & Game Math",
        "icon": "🎮",
        "description": "Render z = f(x,y) as a rotating mesh for terrain and physics",
        "params": {
            "expression": {"type": "text", "default": "sin(x)*cos(y)", "label": "z = f(x,y)"},
            "x_range": {"type": "range", "default": [-3, 3]},
            "y_range": {"type": "range", "default": [-3, 3]},
            "resolution": {"type": "slider", "min": 10, "max": 50, "default": 25},
        },
        "outputs": ["mesh_vertices", "mesh_faces", "gradient_colors"],
        "curriculum_topics": ["3D surface plots"],
    },
    "quaternion_rotation": {
        "name": "Quaternion Rotation Visualizer",
        "category": "3D & Game Math",
        "icon": "🎮",
        "description": "How quaternions rotate objects in 3D without gimbal lock",
        "params": {
            "axis": {"type": "vector", "default": [0, 1, 0], "label": "Rotation axis"},
            "angle": {"type": "slider", "min": 0, "max": 360, "step": 1, "default": 45, "unit": "degrees"},
            "show_gimbal_comparison": {"type": "toggle", "default": True},
        },
        "outputs": ["quaternion", "rotation_matrix", "rotated_object", "euler_comparison"],
        "curriculum_topics": ["Quaternion rotation visualizer"],
    },
    "bezier_curves": {
        "name": "Bezier & Spline Curves",
        "category": "3D & Game Math",
        "icon": "🎮",
        "description": "Drag control points, watch smooth interpolated paths form",
        "params": {
            "control_points": {"type": "point_list", "default": [[0, 0], [1, 3], [3, 3], [4, 0]]},
            "curve_type": {"type": "select", "options": ["quadratic", "cubic", "catmull_rom"], "default": "cubic"},
            "t_resolution": {"type": "slider", "min": 10, "max": 200, "default": 100},
        },
        "outputs": ["curve_points", "tangent_at_t", "curvature"],
        "curriculum_topics": ["Bezier & spline curves"],
    },

    # ── Graph Theory & Discrete Math ──────────────────────────────────────
    "graph_traversal": {
        "name": "Graph Traversal (BFS/DFS)",
        "category": "Graph Theory",
        "icon": "🔗",
        "description": "Animate nodes being explored step-by-step",
        "params": {
            "algorithm": {"type": "select", "options": ["BFS", "DFS"], "default": "BFS"},
            "graph": {"type": "adjacency_list", "default": {"0": ["1", "2"], "1": ["3"], "2": ["3", "4"], "3": [], "4": []}},
            "start_node": {"type": "text", "default": "0"},
            "speed_ms": {"type": "slider", "min": 100, "max": 2000, "step": 100, "default": 500},
        },
        "outputs": ["traversal_order", "step_states", "visited_edges"],
        "curriculum_topics": ["BFS and DFS traversal", "Graph representations (adjacency list/matrix)"],
    },
    "fractal_explorer": {
        "name": "Fractal Explorer",
        "category": "Graph Theory",
        "icon": "🔗",
        "description": "Mandelbrot and Julia sets, procedurally generated",
        "params": {
            "fractal_type": {"type": "select", "options": ["mandelbrot", "julia"], "default": "mandelbrot"},
            "julia_c_real": {"type": "slider", "min": -2, "max": 2, "step": 0.01, "default": -0.7},
            "julia_c_imag": {"type": "slider", "min": -2, "max": 2, "step": 0.01, "default": 0.27015},
            "max_iterations": {"type": "slider", "min": 10, "max": 500, "default": 100},
            "zoom": {"type": "slider", "min": 0.5, "max": 50, "step": 0.5, "default": 1},
        },
        "outputs": ["pixel_data", "iteration_counts"],
        "curriculum_topics": ["Fractals (Mandelbrot/Julia sets)"],
    },
}


def get_visualization_modules() -> dict:
    """Return all available visualization modules grouped by category."""
    by_category = {}
    for mod_id, mod in VISUALIZATION_MODULES.items():
        cat = mod["category"]
        by_category.setdefault(cat, []).append({"id": mod_id} | mod)
    return by_category


def get_module(module_id: str) -> dict:
    """Return a single visualization module definition."""
    mod = VISUALIZATION_MODULES.get(module_id)
    if not mod:
        return {"error": f"Module '{module_id}' not found"}
    return {"id": module_id} | mod
