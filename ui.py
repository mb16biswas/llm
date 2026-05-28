import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Compliance Checker", layout="wide")

st.title("📊 Compliance Assessment Tool")

# --------------------------------------------------
# ✅ Dummy CSV Data (Defined but not necessarily used)
# --------------------------------------------------

# Input Dataset
dummy_input_csv = """Law Id,Law Name,Provision Name,Task Name,Task Description
3030,Energy Conservation Act,Section 7,Energy Policy,Organisation should have an energy policy
3120,Factories Act,Rule 90B,Health and Safety policy,Every factory must have written safety policy
5112,Public Liability Act,Section 4,Public liability policy,Obtain insurance before handling hazardous material
3120,Factories Act,Section 51,Working hours,Max 48 hours per week with rest intervals
"""

# Policy Dataset
dummy_policy_csv = """Policy Name,Policy Text
abc_India_Policy_-_Working_Hours,abc follows 45 hours per week and 5-day work week
abc_India_Policy___Earned_Vacation,Employees get 16 days paid leave annually
abc_India_Safety_Policy,Safety rules must be followed at workplace
"""

# Convert dummy CSVs into DataFrames
dummy_input_df = pd.read_csv(StringIO(dummy_input_csv))
dummy_policy_df = pd.read_csv(StringIO(dummy_policy_csv))

# Show dummy datasets (optional UI toggle)
with st.expander("🔧 View Dummy Data (For Testing)"):
    st.write("### Dummy Input Data")
    st.dataframe(dummy_input_df)

    st.write("### Dummy Policy Data")
    st.dataframe(dummy_policy_df)


# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Input File (Excel/CSV)",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:

    # -------------------------
    # Read File
    # -------------------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # -------------------------
    # Display Input Preview
    # -------------------------
    st.subheader("📥 Input Data Preview")
    st.dataframe(df, use_container_width=True)

    # -------------------------
    # Placeholder Processing Function
    # -------------------------
    def process_data(input_df):
        """
        Placeholder function.
        Replace with real logic later.
        """

        output_df = input_df.copy()

        output_df["Policy Exists"] = ["NO"] * len(output_df)
        output_df["Present"] = [""] * len(output_df)
        output_df["Present but Mismatch"] = ["[]"] * len(output_df)
        output_df["Absent"] = ["[]"] * len(output_df)

        # Dummy logic (for UI demo)
        for i in range(len(output_df)):
            if i % 2 == 0:
                output_df.at[i, "Absent"] = "Missing requirement"
            else:
                output_df.at[i, "Present but Mismatch"] = "Mismatch found"
                output_df.at[i, "Policy Exists"] = "YES"

        return output_df

    # -------------------------
    # Run Processing
    # -------------------------
    st.subheader("⚙️ Processing Data...")

    output_df = process_data(df)

    # -------------------------
    # Filter only issues
    # -------------------------
    filtered_df = output_df[
        (output_df["Present but Mismatch"] != "[]") |
        (output_df["Absent"] != "[]")
    ]

    # -------------------------
    # Display Output
    # -------------------------
    st.subheader("📤 Compliance Issues (Mismatch & Absent Only)")

    if filtered_df.empty:
        st.success("✅ No issues found!")
    else:
        st.dataframe(filtered_df, use_container_width=True)

    # -------------------------
    # Download Option
    # -------------------------
    csv = filtered_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Result",
        data=csv,
        file_name="compliance_issues.csv",
        mime="text/csv",
    )
