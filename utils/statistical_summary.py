import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Union
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend


def create_summary_dict(chunk: pd.DataFrame, level_name: str, level_value: str) -> Dict[str, Any]:
    """Create a structured dictionary with sales summary information
    
    Args:
        chunk: DataFrame containing the data to summarize
        level_name: Name of the aggregation level (e.g., "product", "region")
        level_value: Value of the aggregation level (e.g., "Widget A", "North")
        
    Returns:
        Dictionary containing summarized sales information
    """
    summary = {
        "level": level_name,
        "value": level_value,
        "total_records": len(chunk),
        "total_sales": float(chunk["Sales"].sum()),
        "average_sale": float(chunk["Sales"].mean()),
        "average_satisfaction": float(chunk["Customer_Satisfaction"].mean()),
        "date_range": {
            "start": chunk["Date"].min().strftime("%Y-%m-%d"),
            "end": chunk["Date"].max().strftime("%Y-%m-%d")
        },
        "product_summary": chunk.groupby("Product")["Sales"].agg(["sum", "count", "mean"]).to_dict("index"),
        "region_summary": chunk.groupby("Region")["Sales"].agg(["sum", "count", "mean"]).to_dict("index"),
        "gender_summary": chunk.groupby("Customer_Gender")["Sales"].agg(["sum", "count", "mean"]).to_dict("index")
    }
    return summary


def create_time_period_plot(chunk: pd.DataFrame, yq: str, plots_dir: str) -> str:
    """Create plot for time period data
    
    Args:
        chunk: DataFrame containing data for the specified time period
        yq: Year-quarter string (e.g., "2023-Q1")
        plots_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    # Set up the figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Monthly sales trend
    chunk_by_month = chunk.set_index("Date").resample("M")["Sales"].sum()
    axs[0, 0].plot(chunk_by_month.index, chunk_by_month.values, marker="o", linewidth=2)
    axs[0, 0].set_title(f"Monthly Sales Trend for {yq}")
    axs[0, 0].set_xlabel("Month")
    axs[0, 0].set_ylabel("Total Sales")
    axs[0, 0].grid(True)
    
    # 2. Product distribution
    product_sales = chunk.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    axs[0, 1].bar(product_sales.index, product_sales.values, color="teal")
    axs[0, 1].set_title(f"Product Sales Distribution for {yq}")
    axs[0, 1].set_xlabel("Product")
    axs[0, 1].set_ylabel("Total Sales")
    plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha="right")
    
    # 3. Regional sales
    region_sales = chunk.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    axs[1, 0].bar(region_sales.index, region_sales.values, color="purple")
    axs[1, 0].set_title(f"Regional Sales for {yq}")
    axs[1, 0].set_xlabel("Region")
    axs[1, 0].set_ylabel("Total Sales")
    
    # 4. Customer satisfaction by product
    satisfaction = chunk.groupby("Product")["Customer_Satisfaction"].mean().sort_values(ascending=False)
    axs[1, 1].bar(satisfaction.index, satisfaction.values, color="orange")
    axs[1, 1].set_title(f"Customer Satisfaction by Product for {yq}")
    axs[1, 1].set_xlabel("Product")
    axs[1, 1].set_ylabel("Average Satisfaction (1-5)")
    plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"time_period_{yq.replace('-', '_')}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path


def create_product_plot(chunk: pd.DataFrame, product: str, plots_dir: str) -> str:
    """Create plot for product data
    
    Args:
        chunk: DataFrame containing data for the specified product
        product: Product name (e.g., "Widget A")
        plots_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sales trend over time
    sales_trend = chunk.set_index("Date").resample("M")["Sales"].sum()
    axs[0, 0].plot(sales_trend.index, sales_trend.values, marker="o", linewidth=2, color="blue")
    axs[0, 0].set_title(f"Monthly Sales Trend for {product}")
    axs[0, 0].set_xlabel("Month")
    axs[0, 0].set_ylabel("Total Sales")
    axs[0, 0].grid(True)
    
    # 2. Regional distribution
    region_sales = chunk.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    axs[0, 1].bar(region_sales.index, region_sales.values, color="green")
    axs[0, 1].set_title(f"Sales by Region for {product}")
    axs[0, 1].set_xlabel("Region")
    axs[0, 1].set_ylabel("Total Sales")
    
    # 3. Sales by age group
    if "Age_Group" in chunk.columns:
        age_sales = chunk.groupby("Age_Group")["Sales"].sum()
        axs[1, 0].bar(age_sales.index, age_sales.values, color="red")
        axs[1, 0].set_title(f"Sales by Age Group for {product}")
        axs[1, 0].set_xlabel("Age Group")
        axs[1, 0].set_ylabel("Total Sales")
    
    # 4. Sales by gender
    gender_sales = chunk.groupby("Customer_Gender")["Sales"].sum()
    axs[1, 1].pie(gender_sales.values, labels=gender_sales.index, autopct="%1.1f%%", 
                  colors=["lightblue", "pink"], startangle=90)
    axs[1, 1].set_title(f"Sales by Gender for {product}")
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"product_{product.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path


def create_region_plot(chunk: pd.DataFrame, region: str, plots_dir: str) -> str:
    """Create plot for region data
    
    Args:
        chunk: DataFrame containing data for the specified region
        region: Region name (e.g., "North", "South")
        plots_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sales trend over time
    sales_trend = chunk.set_index("Date").resample("M")["Sales"].sum()
    axs[0, 0].plot(sales_trend.index, sales_trend.values, marker="o", linewidth=2, color="green")
    axs[0, 0].set_title(f"Monthly Sales Trend for {region} Region")
    axs[0, 0].set_xlabel("Month")
    axs[0, 0].set_ylabel("Total Sales")
    axs[0, 0].grid(True)
    
    # 2. Product distribution
    product_sales = chunk.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    axs[0, 1].bar(product_sales.index, product_sales.values, color="blue")
    axs[0, 1].set_title(f"Product Sales Distribution for {region} Region")
    axs[0, 1].set_xlabel("Product")
    axs[0, 1].set_ylabel("Total Sales")
    plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha="right")
    
    # 3. Customer satisfaction by product
    satisfaction = chunk.groupby("Product")["Customer_Satisfaction"].mean().sort_values(ascending=False)
    axs[1, 0].bar(satisfaction.index, satisfaction.values, color="orange")
    axs[1, 0].set_title(f"Customer Satisfaction by Product in {region} Region")
    axs[1, 0].set_xlabel("Product")
    axs[1, 0].set_ylabel("Average Satisfaction (1-5)")
    plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha="right")
    
    # 4. Sales by gender
    gender_sales = chunk.groupby("Customer_Gender")["Sales"].sum()
    axs[1, 1].pie(gender_sales.values, labels=gender_sales.index, autopct="%1.1f%%",
                  colors=["lightblue", "pink"], startangle=90)
    axs[1, 1].set_title(f"Sales by Gender for {region} Region")
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"region_{region}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path


def create_age_group_plot(chunk: pd.DataFrame, age_group: Union[str, pd.Interval], plots_dir: str) -> str:
    """Create plot for age group data
    
    Args:
        chunk: DataFrame containing data for the specified age group
        age_group: Age group label (e.g., "18-25")
        plots_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sales trend over time
    sales_trend = chunk.set_index("Date").resample("M")["Sales"].sum()
    axs[0, 0].plot(sales_trend.index, sales_trend.values, marker="o", linewidth=2, color="purple")
    axs[0, 0].set_title(f"Monthly Sales Trend for Age Group {age_group}")
    axs[0, 0].set_xlabel("Month")
    axs[0, 0].set_ylabel("Total Sales")
    axs[0, 0].grid(True)
    
    # 2. Product preference
    product_sales = chunk.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    axs[0, 1].bar(product_sales.index, product_sales.values, color="teal")
    axs[0, 1].set_title(f"Product Preference for Age Group {age_group}")
    axs[0, 1].set_xlabel("Product")
    axs[0, 1].set_ylabel("Total Sales")
    plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha="right")
    
    # 3. Regional distribution
    region_sales = chunk.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    axs[1, 0].bar(region_sales.index, region_sales.values, color="green")
    axs[1, 0].set_title(f"Sales by Region for Age Group {age_group}")
    axs[1, 0].set_xlabel("Region")
    axs[1, 0].set_ylabel("Total Sales")
    
    # 4. Sales by gender with satisfaction
    if len(chunk["Customer_Gender"].unique()) > 1:
        gender_sat = chunk.groupby("Customer_Gender")["Customer_Satisfaction"].mean()
        gender_sales = chunk.groupby("Customer_Gender")["Sales"].sum()
        
        ax4 = axs[1, 1]
        x = range(len(gender_sat))
        width = 0.35
        
        ax4.bar([i - width/2 for i in x], gender_sales.values, width, label="Sales", color="blue")
        
        ax_right = ax4.twinx()
        ax_right.bar([i + width/2 for i in x], gender_sat.values, width, label="Satisfaction", color="red", alpha=0.7)
        ax_right.set_ylim(0, 5)
        ax_right.set_ylabel("Satisfaction (1-5)")
        
        ax4.set_title(f"Sales and Satisfaction by Gender for Age Group {age_group}")
        ax4.set_xticks(x)
        ax4.set_xticklabels(gender_sat.index)
        ax4.set_ylabel("Total Sales")
        
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    else:
        gender = chunk["Customer_Gender"].iloc[0]
        axs[1, 1].text(0.5, 0.5, f"Only {gender} customers in this age group", 
                       horizontalalignment="center", verticalalignment="center", transform=axs[1, 1].transAxes)
        axs[1, 1].set_title(f"Gender Distribution for Age Group {age_group}")
    
    plt.tight_layout()
    
    # Save the figure
    safe_age_group = str(age_group).replace("-", "to")
    plot_path = os.path.join(plots_dir, f"age_group_{safe_age_group}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path


def create_overall_plot(df: pd.DataFrame, plots_dir: str) -> str:
    """Create plot for overall data summary
    
    Args:
        df: DataFrame containing all sales data
        plots_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall sales trend
    sales_trend = df.set_index("Date").resample("M")["Sales"].sum()
    axs[0, 0].plot(sales_trend.index, sales_trend.values, marker="o", linewidth=2, color="blue")
    axs[0, 0].set_title("Overall Monthly Sales Trend")
    axs[0, 0].set_xlabel("Month")
    axs[0, 0].set_ylabel("Total Sales")
    axs[0, 0].grid(True)
    
    # 2. Sales distribution by product
    product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    axs[0, 1].bar(product_sales.index, product_sales.values, color="teal")
    axs[0, 1].set_title("Total Sales by Product")
    axs[0, 1].set_xlabel("Product")
    axs[0, 1].set_ylabel("Total Sales")
    plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha="right")
    
    # 3. Sales by region
    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    axs[1, 0].bar(region_sales.index, region_sales.values, color="purple")
    axs[1, 0].set_title("Total Sales by Region")
    axs[1, 0].set_xlabel("Region")
    axs[1, 0].set_ylabel("Total Sales")
    
    # 4. Customer satisfaction overview
    if "Age_Group" in df.columns:
        pivot = df.pivot_table(
            index="Age_Group", 
            columns="Customer_Gender", 
            values="Customer_Satisfaction", 
            aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=axs[1, 1])
        axs[1, 1].set_title("Customer Satisfaction by Age Group and Gender")
        axs[1, 1].set_xlabel("Gender")
        axs[1, 1].set_ylabel("Age Group")
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, "overall_summary.png")
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path

