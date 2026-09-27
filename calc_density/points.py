# points.py


def get_plot_points(
    x,
    f,
    f_min,
    f_max,
    k,
    theta,
    N,
    save_points=100
):
    indices = [
        round(
            i * (len(x) - 1) / (save_points - 1)
        )
        for i in range(save_points)
    ]

    points = []

    for i in indices:
        points.append({
            "x": float(x[i]),
            "f": float(f[i]),
            "f_min": float(f_min[i]),
            "f_max": float(f_max[i])
        })

    return {
        "k": k,
        "theta": theta,
        "N": N,
        "points": points
    }