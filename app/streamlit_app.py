# Streamlit Application

import streamlit as st


def main():
    # Other sections...
    # Data Snapshot
    st.dataframe(filtered.head(10).astype(str), use_container_width=True)

if __name__ == '__main__':
    main()