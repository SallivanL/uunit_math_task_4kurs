import os

from logger import log

AREA_FILE = "results/calc_areas.md"

def init_area_report():
    os.makedirs("results", exist_ok=True)

    with open(AREA_FILE, "w", encoding="utf-8") as f:
        f.write("# Расчёт площадей под кривыми\n\n")
        f.write(
            "| k | theta | N | m | функция | "
            "полная площадь | площадь m | доля | "
            "x_left | x_peak | x_right |\n"
        )
        f.write(
            "|---:|---:|---:|---:|---|---:|---:|"
            "---:|---:|---:|---:|\n"
        )


def save_area_row(
    k,
    theta,
    N,
    m,
    function_name,
    area_info
):
    os.makedirs("results", exist_ok=True)

    with open(
        AREA_FILE,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(
            f"| {k} | {theta} | {N} | {m} | "
            f"{function_name} | "
            f"{area_info['total_area']:.10f} | "
            f"{area_info['area']:.10f} | "
            f"{area_info['ratio']:.10f} | "
            f"{area_info['left_x']:.10f} | "
            f"{area_info['peak_x']:.10f} | "
            f"{area_info['right_x']:.10f} |\n"
        )

    log(
        "Расчёт площади",
        k=k,
        theta=theta,
        N=N,
        m=m,
        function=function_name,
        total_area=area_info["total_area"],
        area=area_info["area"],
        ratio=area_info["ratio"],
        x_left=area_info["left_x"],
        x_peak=area_info["peak_x"],
        x_right=area_info["right_x"]
    )