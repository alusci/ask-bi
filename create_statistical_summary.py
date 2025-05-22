import argparse
import pandas as pd
import json
import os
import warnings
from utils.statistical_summary import (
    create_summary_dict,
    create_time_period_plot,
    create_product_plot,
    create_region_plot,
    create_age_group_plot,
    create_overall_plot
)

# Filter out warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create statistical summaries for sales data")
    parser.add_argument(
        "--input", 
        type=str,
        default="Datasets/sales_data.csv", 
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="sales_data_statistics",
        help="Path to the output directory"
    )
    
    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file
    Args:
        file_path (str): Path to the CSV file
    Returns:
        pd.DataFrame: DataFrame containing the loaded data
    """
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Extract time components
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["YearQuarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)

    # Create age groups for customer segmentation
    df["Age_Group"] = pd.cut(
        df["Customer_Age"], 
        bins=[17, 25, 35, 50, 70], 
        labels=["18-25", "26-35", "36-50", "51-70"]
    )

    return df


if __name__ == "__main__":
    args = parse_args()
    
    sales_df = load_data(args.input)

    # Create output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    plots_dir = os.path.join(args.output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create documents for different dimensions of analysis
    documents = []

    # 1. Time period documents (by year-quarter)
    for yq in sales_df["YearQuarter"].unique():
        chunk = sales_df[sales_df["YearQuarter"] == yq]
        summary = create_summary_dict(chunk, "time_period", yq)
        
        # Create and save plot
        plot_path = create_time_period_plot(chunk, yq, plots_dir)
        
        # Convert to text format suitable for embeddings
        text = f"Sales Summary for {yq}\n"
        text += f"Total Records: {summary['total_records']}\n"
        text += f"Total Sales: ${summary['total_sales']:.2f}\n"
        text += f"Average Sale: ${summary['average_sale']:.2f}\n"
        text += f"Average Customer Satisfaction: {summary['average_satisfaction']:.2f}\n"
        text += f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
        
        text += "Sales by Product:\n"
        for product, stats in summary["product_summary"].items():
            text += f"- {product}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        text += "\nSales by Region:\n"
        for region, stats in summary["region_summary"].items():
            text += f"- {region}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        documents.append({
            "id": f"time_{yq}",
            "text": text,
            "metadata": {
                "type": "time_period",
                "period": yq,
                "raw_data": summary,
                "plot_path": plot_path
            }
        })

    # 2. Product documents
    for product in sales_df["Product"].unique():
        chunk = sales_df[sales_df["Product"] == product]
        summary = create_summary_dict(chunk, "product", product)
        
        # Create and save plot
        plot_path = create_product_plot(chunk, product, plots_dir)
        
        text = f"Sales Summary for {product}\n"
        text += f"Total Records: {summary['total_records']}\n"
        text += f"Total Sales: ${summary['total_sales']:.2f}\n"
        text += f"Average Sale: ${summary['average_sale']:.2f}\n"
        text += f"Average Customer Satisfaction: {summary['average_satisfaction']:.2f}\n"
        text += f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
        
        text += "Sales by Region:\n"
        for region, stats in summary["region_summary"].items():
            text += f"- {region}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        text += "\nSales by Gender:\n"
        for gender, stats in summary["gender_summary"].items():
            text += f"- {gender}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        documents.append({
            "id": f"product_{product.replace(' ', '_')}",
            "text": text,
            "metadata": {
                "type": "product",
                "product": product,
                "raw_data": summary,
                "plot_path": plot_path
            }
        })

    # 3. Region documents
    for region in sales_df["Region"].unique():
        chunk = sales_df[sales_df["Region"] == region]
        summary = create_summary_dict(chunk, "region", region)
        
        # Create and save plot
        plot_path = create_region_plot(chunk, region, plots_dir)
        
        text = f"Sales Summary for {region} Region\n"
        text += f"Total Records: {summary['total_records']}\n"
        text += f"Total Sales: ${summary['total_sales']:.2f}\n"
        text += f"Average Sale: ${summary['average_sale']:.2f}\n"
        text += f"Average Customer Satisfaction: {summary['average_satisfaction']:.2f}\n"
        text += f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
        
        text += "Sales by Product:\n"
        for product, stats in summary["product_summary"].items():
            text += f"- {product}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        text += "\nSales by Gender:\n"
        for gender, stats in summary["gender_summary"].items():
            text += f"- {gender}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        documents.append({
            "id": f"region_{region}",
            "text": text,
            "metadata": {
                "type": "region",
                "region": region,
                "raw_data": summary,
                "plot_path": plot_path
            }
        })

    # 4. Demographic documents (by age group)
    for age_group in sales_df["Age_Group"].unique():
        chunk = sales_df[sales_df["Age_Group"] == age_group]
        summary = create_summary_dict(chunk, "age_group", str(age_group))
        
        # Create and save plot
        safe_age_group = str(age_group).replace("-", "to")
        plot_path = create_age_group_plot(chunk, age_group, plots_dir)
        
        text = f"Sales Summary for Age Group {age_group}\n"
        text += f"Total Records: {summary['total_records']}\n"
        text += f"Total Sales: ${summary['total_sales']:.2f}\n"
        text += f"Average Sale: ${summary['average_sale']:.2f}\n"
        text += f"Average Customer Satisfaction: {summary['average_satisfaction']:.2f}\n"
        text += f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
        
        text += "Sales by Product:\n"
        for product, stats in summary["product_summary"].items():
            text += f"- {product}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        text += "\nSales by Region:\n"
        for region, stats in summary["region_summary"].items():
            text += f"- {region}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        text += "\nSales by Gender:\n"
        for gender, stats in summary["gender_summary"].items():
            text += f"- {gender}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"
        
        documents.append({
            "id": f"age_group_{str(age_group)}",
            "text": text,
            "metadata": {
                "type": "demographic",
                "age_group": str(age_group),
                "raw_data": summary,
                "plot_path": plot_path
            }
        })

    # 5. Overall statistics document
    overall_summary = create_summary_dict(sales_df, "overall", "all_data")
    
    # Create and save plot
    plot_path = create_overall_plot(sales_df, plots_dir)

    text = "Overall Sales Summary\n"
    text += f"Total Records: {overall_summary['total_records']}\n"
    text += f"Total Sales: ${overall_summary['total_sales']:.2f}\n"
    text += f"Average Sale: ${overall_summary['average_sale']:.2f}\n"
    text += f"Average Customer Satisfaction: {overall_summary['average_satisfaction']:.2f}\n"
    text += f"Date Range: {overall_summary['date_range']['start']} to {overall_summary['date_range']['end']}\n\n"

    text += "Sales by Product:\n"
    for product, stats in overall_summary["product_summary"].items():
        text += f"- {product}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"

    text += "\nSales by Region:\n"
    for region, stats in overall_summary["region_summary"].items():
        text += f"- {region}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"

    text += "\nSales by Gender:\n"
    for gender, stats in overall_summary["gender_summary"].items():
        text += f"- {gender}: ${stats['sum']:.2f} total, {stats['count']} sales, ${stats['mean']:.2f} average\n"

    documents.append({
        "id": "overall_summary",
        "text": text,
        "metadata": {
            "type": "overall",
            "raw_data": overall_summary,
            "plot_path": plot_path
        }
    })

    # Save the documents
    with open(f"{args.output_dir}/documents.json", "w") as f:
        json.dump(documents, f, indent=2)

    print(f"Created {len(documents)} documents with visualizations for the RAG system")
    print(f"Plots saved in {plots_dir}")
