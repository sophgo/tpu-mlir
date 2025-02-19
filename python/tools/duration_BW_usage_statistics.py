import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def find_Start_Cycle_row_index_by_keyword(file_path, keyword, num_core):
    xl = pd.ExcelFile(file_path)
    duration_sheet_name = xl.sheet_names[1] if num_core != 2 else xl.sheet_names[3]
    bandwidth_usage_sheet_name = (
        xl.sheet_names[2] if num_core != 2 else xl.sheet_names[4]
    )
    df_duration = xl.parse(duration_sheet_name)
    df_bandwidth_usage = xl.parse(bandwidth_usage_sheet_name)
    for index, row in df_duration.iterrows():
        if row.astype(str).str.contains(keyword, regex=False).any():
            TIU_Start_Cycle_row_index = int(index) + 1
    for index_, row_ in df_bandwidth_usage.iterrows():
        if row_.astype(str).str.contains(keyword, regex=False).any():
            DMA_Start_Cycle_row_index = int(index_) + 1

    return TIU_Start_Cycle_row_index, DMA_Start_Cycle_row_index


def read_perfAI_excel(perfAI_excel_path, num_core):
    df_second_sheet = pd.read_excel(perfAI_excel_path, sheet_name=1)
    tiu_frequency = int(str(df_second_sheet["TIU Frequency(MHz)"].dropna().iloc[0]))
    dma_frequency = int(str(df_second_sheet["DMA Frequency(MHz)"].dropna().iloc[0]))

    (
        TIU_Start_Cycle_row_index,
        DMA_Start_Cycle_row_index,
    ) = find_Start_Cycle_row_index_by_keyword(
        perfAI_excel_path, "Start Cycle", num_core
    )
    duration_sheet_index = 1 if num_core != 2 else 3
    bandwidth_usage_sheet_index = 2 if num_core != 2 else 4

    duration_data_df = pd.read_excel(
        perfAI_excel_path,
        sheet_name=duration_sheet_index,
        skiprows=TIU_Start_Cycle_row_index,
        usecols=["Cmd Id", "Start Cycle", "End Cycle"],
        dtype={"Cmd Id": int, "Start Cycle": float, "End Cycle": float},
    )

    bandwidth_usage_data_df = pd.read_excel(
        perfAI_excel_path,
        sheet_name=bandwidth_usage_sheet_index,
        skiprows=DMA_Start_Cycle_row_index,
        usecols=["Cmd Id", "DMA data size(B)", "Start Cycle", "End Cycle"],
        dtype={
            "Cmd Id": int,
            "DMA data size(B)": float,
            "Start Cycle": float,
            "End Cycle": float,
        },
    )

    return duration_data_df, bandwidth_usage_data_df, tiu_frequency, dma_frequency


def read_mlir_json(
    mlir_json_path,
    duration_data_df,
    bandwidth_usage_data_df,
    tiu_frequency,
    dma_frequency,
):
    with open(mlir_json_path, "r") as f:
        data = json.load(f)

    csv_data = [
        [
            "File Line",
            "Opcode",
            "TIU DMA ID (Before)",
            "TIU DMA ID (After)",
            "TIU ID",
            "DMA ID",
        ]
    ]
    duration_dict = {}
    bandwidth_usage_dict = {}
    opname_dict = {}

    for items in data:
        file_line = items.get("file-line")
        opcode = items.get("opcode")
        tiu_dma_id_before = items.get("tiu_dma_id(before)")
        tiu_dma_id_after = items.get("tiu_dma_id(after)")
        tiu_id = [tiu_dma_id_before[0], tiu_dma_id_after[0]]
        dma_id = [tiu_dma_id_before[1], tiu_dma_id_after[1]]
        csv_data.append(
            [file_line, opcode, tiu_dma_id_before, tiu_dma_id_after, tiu_id, dma_id]
        )

        if not (
            tiu_id[0] + 1 < len(duration_data_df) and tiu_id[1] < len(duration_data_df)
        ):
            continue
        if not (
            dma_id[0] + 1 < len(bandwidth_usage_data_df)
            and dma_id[1] < len(bandwidth_usage_data_df)
        ):
            continue

        filtered_duration_df1 = duration_data_df[
            duration_data_df["Cmd Id"] == tiu_id[0] + 1
        ]
        filtered_duration_df2 = duration_data_df[
            duration_data_df["Cmd Id"] == tiu_id[1]
        ]

        if not filtered_duration_df1.empty and not filtered_duration_df2.empty:
            tiu_start_cycle_left = float(filtered_duration_df1["Start Cycle"].values[0])
            tiu_end_cycle_right = float(filtered_duration_df2["End Cycle"].values[0])
        else:
            continue

        filtered_BW_usage_df1 = bandwidth_usage_data_df[
            bandwidth_usage_data_df["Cmd Id"] == dma_id[0] + 1
        ]
        filtered_BW_usage_df2 = bandwidth_usage_data_df[
            bandwidth_usage_data_df["Cmd Id"] == dma_id[1]
        ]

        if not filtered_BW_usage_df1.empty and not filtered_BW_usage_df2.empty:
            dma_start_cycle_left = float(filtered_BW_usage_df1["Start Cycle"].values[0])
            dma_end_cycle_right = float(filtered_BW_usage_df2["End Cycle"].values[0])
        else:
            continue

        bandwidth_usage_op = calculate_bandwidth_usage(dma_id, bandwidth_usage_data_df)

        duration_op = calculate_duration(
            tiu_id,
            dma_id,
            tiu_start_cycle_left,
            tiu_end_cycle_right,
            dma_start_cycle_left,
            dma_end_cycle_right,
            tiu_frequency,
            dma_frequency,
        )

        duration_dict[file_line] = duration_dict.get(file_line, 0) + duration_op
        bandwidth_usage_dict[file_line] = (
            bandwidth_usage_dict.get(file_line, 0) + bandwidth_usage_op
        )

        sorted_duration = sorted(duration_dict.items(), key=lambda x: x[0])
        sorted_bandwidth_usage = sorted(
            bandwidth_usage_dict.items(), key=lambda x: x[0]
        )
        opname_dict[file_line] = opcode

    json_message_csv_file_output_path = "json_message_output.csv"
    with open(json_message_csv_file_output_path, "w") as csv_file:
        for row in csv_data:
            csv_file.write("\t".join(map(str, row)) + "\n")

    return sorted_duration, sorted_bandwidth_usage, opname_dict


def calculate_duration(
    tiu_id,
    dma_id,
    tiu_start_cycle_left,
    tiu_end_cycle_right,
    dma_start_cycle_left,
    dma_end_cycle_right,
    tiu_frequency,
    dma_frequency,
):
    if tiu_id[0] == tiu_id[1] and dma_id[0] == dma_id[1]:
        return 0
    elif tiu_id[0] == tiu_id[1]:
        return (dma_end_cycle_right - dma_start_cycle_left) / dma_frequency * 1000
    elif dma_id[0] == dma_id[1]:
        return (tiu_end_cycle_right - tiu_start_cycle_left) / tiu_frequency * 1000
    else:
        merge_interval_left = (
            min(
                tiu_start_cycle_left / tiu_frequency,
                dma_start_cycle_left / dma_frequency,
            )
            * 1000
        )
        merge_interval_right = (
            max(
                tiu_end_cycle_right / tiu_frequency, dma_end_cycle_right / dma_frequency
            )
            * 1000
        )
        return merge_interval_right - merge_interval_left


def calculate_bandwidth_usage(dma_id, bandwidth_usage_data_df):
    return (
        bandwidth_usage_data_df.loc[
            (bandwidth_usage_data_df["Cmd Id"] >= dma_id[0] + 1)
            & (bandwidth_usage_data_df["Cmd Id"] <= dma_id[1]),
            "DMA data size(B)",
        ]
        .astype(float)
        .sum()
    )


def write_csv_data_duration(
    output_duration_CSV_file,
    sorted_duration,
    sorted_bandwidth_usage,
    opname_dict,
    num_core,
):
    x_labels = []
    y_duration_values = []
    y_bandwidth_usage_values = []
    op_labels = []
    y_duration_percentage = []
    y_BW_usage_percentage = []

    total_duration = sum([item[1] for item in sorted_duration])
    total_BW_usage = sum([item_[1] for item_ in sorted_bandwidth_usage])

    with open(
        f"{num_core}_{output_duration_CSV_file}", "w", encoding="UTF-8"
    ) as csv_file:
        csv_file.write(
            f"{'File Line':<10} {'Operation_Name':<20} {'Duration(ns)':<20} {'Bandwidth_Usage(B)':<20}\n"
        )
        for (file_line, duration), (_, bandwidth_usage) in zip(
            sorted_duration, sorted_bandwidth_usage
        ):
            x_labels.append(str(file_line))
            y_duration_values.append(duration)
            op_labels.append(str(opname_dict[file_line]).replace("tpu.", ""))
            y_bandwidth_usage_values.append(round(bandwidth_usage, 1))
            duration_percentage = duration / total_duration * 100
            y_duration_percentage.append(str(round(duration_percentage, 1)) + "%")
            BW_usage_percentage = bandwidth_usage / total_BW_usage * 100
            y_BW_usage_percentage.append(str(round(BW_usage_percentage, 1)) + "%")
            csv_file.write(
                f"{file_line:<10} {opname_dict[file_line]:<20} {round(duration, 1):<20} {round(bandwidth_usage, 1):<20}\n"
            )

    plot_duration_and_bandwidth_usage_chart(
        x_labels,
        op_labels,
        "Duration(ns)",
        "Bandwidth_Usage(B)",
        y_duration_values,
        y_bandwidth_usage_values,
        num_core,
        y_duration_percentage,
        y_BW_usage_percentage,
    )


def plot_duration_and_bandwidth_usage_chart(
    x_labels,
    op_labels,
    keyword_A,
    keyword_B,
    A_values,
    B_values,
    num_core,
    A_percentage=None,
    B_percentage=None,
):
    fig = plt.figure(figsize=(300, 60))
    ax1 = fig.add_axes([0.1, 0.463, 0.8, 0.35])
    bars1 = ax1.bar(x_labels, A_values, color="orange")
    ax1.set_ylabel(keyword_A, fontsize=60)
    ax1.tick_params(axis="y", labelsize=60)
    ax1.yaxis.get_offset_text().set_fontsize(50)
    _, xmax = ax1.get_xlim()
    ax1.yaxis.grid(True, linestyle="--", linewidth=1, color="gray")
    ax1.text(xmax, 0, "———   file_line   ———", fontsize=40, ha="right", va="top")

    ax1.tick_params(axis="x", labelbottom=False)
    for bar, label in zip(bars1, op_labels):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=30,
            rotation=45,
        )

    if A_percentage:
        for bar, label in zip(bars1, A_percentage):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                label,
                ha="center",
                va="bottom",
                fontsize=20,
                rotation=45,
                color="m",
            )

    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.35])
    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels, fontsize=20, rotation=45, color="b")
    bars2 = ax2.bar(x_labels, B_values, color="g")
    ax2.set_ylabel(keyword_B, fontsize=60)
    ax2.tick_params(axis="y", labelsize=60)
    ax2.yaxis.get_offset_text().set_fontsize(50)
    ax2.yaxis.grid(True, linestyle="--", linewidth=1, color="gray")
    ax2.xaxis.tick_top()
    ax2.invert_yaxis()
    ax2.yaxis.get_offset_text().set_visible(False)

    if B_percentage:
        for bar, label in zip(bars2, B_percentage):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                label,
                ha="center",
                va="top",
                fontsize=20,
                rotation=45,
                color="black",
            )
    plt.savefig(
        str(num_core)
        + "_core_"
        + str(keyword_A)
        + "_and_"
        + str(keyword_B)
        + "_chart.png"
    )


def workflow(perfAI_excel_path, mlir_json_path, output_duration_CSV_file, num_core):
    (
        duration_data_df,
        bandwidth_usage_data_df,
        tiu_frequency,
        dma_frequency,
    ) = read_perfAI_excel(perfAI_excel_path, num_core)
    sorted_duration, sorted_bandwidth, opname_dict = read_mlir_json(
        mlir_json_path,
        duration_data_df,
        bandwidth_usage_data_df,
        tiu_frequency,
        dma_frequency,
    )
    write_csv_data_duration(
        output_duration_CSV_file,
        sorted_duration,
        sorted_bandwidth,
        opname_dict,
        num_core,
    )
    print("{}_core workflow succeeded".format(num_core))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the workflow with given parameters."
    )
    parser.add_argument(
        "--mlir_json", type=str, required=True, help="Path to the MLIR JSON file."
    )
    parser.add_argument(
        "--perfai_excel", type=str, required=True, help="Path to the PerfAI Excel file."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="duration_and_BW_usage_output.csv",
        help='Path to the output CSV file. Optional, defaults to "output_duration_and_BW_usage.csv".',
    )

    args = parser.parse_args()

    sheet_to_cores_map = {3: [1], 5: [1, 2], "other": [1, 2, 8]}
    x = pd.ExcelFile(args.perfai_excel)
    num_sheets = len(x.sheet_names)
    num_core_list = sheet_to_cores_map.get(num_sheets) or sheet_to_cores_map["other"]

    for num_core in num_core_list:
        workflow(args.perfai_excel, args.mlir_json, args.output_csv, num_core)
