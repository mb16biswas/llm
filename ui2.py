import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Compliance Checker", layout="wide")

st.title("📊 Compliance Assessment Tool")

# --------------------------------------------------
# ✅ Dummy INPUT CSV (FULL SCHEMA — NO COLUMN DROPPED)
# --------------------------------------------------
dummy_input_csv = """Law Id,Law Name,Corporate Function,Jurisdiction Name,Country,Provision Id,Provision Name,Task Id,Task Name,Task Description,Task Helping Hand
3030,Energy Conservation Act,Administration,Kerala,India,389,Section 7,386,Energy Policy,Organisation should have an Energy policy,
3120,Factories Act,Administration,Goa,India,494,Rule 90B,668,Health and Safety policy,Every factory must have written safety policy,
5112,Public Liability Act,Administration,India,India,6178,Section 4,7424,Public liability policy,Obtain insurance before handling hazardous material,
3120,Factories Act,Administration,Goa,India,8450,Section 51,10231,Working hours,Max 48 hours per week with rest intervals,
"""

# --------------------------------------------------
# ✅ Dummy POLICY CSV
# --------------------------------------------------
dummy_policy_csv = """Policy Name,Policy Text
abc_India_Policy_-_Working_Hours,abc follows 45 hours per week and 5-day work week
abc_India_Policy___Earned_Vacation,Employees get 16 days paid leave annually
abc_India_Safety_Policy,Safety rules must be followed at workplace
"""

dummy_input_df = pd.read_csv(StringIO(dummy_input_csv))
dummy_policy_df = pd.read_csv(StringIO(dummy_policy_csv))

# Show dummy data
with st.expander("🔧 View Dummy Data"):
    st.write("### Input Data (Full Schema)")
    st.dataframe(dummy_input_df)

    st.write("### Policy Data")
    st.dataframe(dummy_policy_df)

# --------------------------------------------------
# Upload File
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Input File (Excel/CSV)",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ✅ Ensure schema is preserved
    st.subheader("📥 Input Data Preview")
    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------
    # ✅ Processing Function (Schema-safe)
    # --------------------------------------------------
    def process_data(input_df):
        """
        Keeps ALL input columns intact.
        Only appends new output columns.
        """

        output_df = input_df.copy()

        # ✅ Add new columns WITHOUT removing existing ones
        output_columns = [
            "Policy Exists",
            "Matched Policies",
            "Present",
            "Present but Mismatch",
            "Absent"
        ]

        for col in output_columns:
            if col not in output_df.columns:
                output_df[col] = ""

        # ✅ Dummy logic (just for UI)
        for i in range(len(output_df)):

            if i % 3 == 0:
                output_df.at[i, "Policy Exists"] = "NO"
                output_df.at[i, "Absent"] = "[Missing policy]"

            elif i % 3 == 1:
                output_df.at[i, "Policy Exists"] = "YES"
                output_df.at[i, "Matched Policies"] = "abc_Safety_Policy"
                output_df.at[i, "Present but Mismatch"] = "[Mismatch in requirements]"

            else:
                output_df.at[i, "Policy Exists"] = "YES"
                output_df.at[i, "Matched Policies"] = "abc_Working_Hours"
                output_df.at[i, "Present"] = "[Fully compliant]"

        return output_df

    # --------------------------------------------------
    # Run Processing
    # --------------------------------------------------
    st.subheader("⚙️ Processing Data...")
    output_df = process_data(df)

    # --------------------------------------------------
    # ✅ Filter ONLY: Mismatch + Absent
    # --------------------------------------------------
    filtered_df = output_df[
        (output_df["Present but Mismatch"] != "") |
        (output_df["Absent"] != "")
    ]

    # --------------------------------------------------
    # Output Display
    # --------------------------------------------------
    st.subheader("📤 Compliance Issues (Mismatch & Absent Only)")

    if filtered_df.empty:
        st.success("✅ No issues found!")
    else:
        st.dataframe(filtered_df, use_container_width=True)

    # --------------------------------------------------
    # Download
    # --------------------------------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Result",
        data=csv,
        file_name="compliance_issues.csv",
        mime="text/csv"
    )
