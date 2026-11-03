import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process files") and uploaded:
        with st.spinner("Extracting, chunking and embedding..."):
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
            try:
                r = requests.post(f"{API}/upload", files=files, timeout=600)
                for item in r.json().get("results", []):
                    if "error" in item:
                        st.error(f"{item['file']}: {item['error']}")
                    else:
                        st.success(f"{item['file']}: {item['pages']} pages, {item['chunks']} chunks")
            except requests.exceptions.ConnectionError:
                st.error("Backend not running. Start it with: uvicorn app.main:app --reload")

    st.divider()
    try:
        docs = requests.get(f"{API}/documents", timeout=10).json()["documents"]
        if docs:
            for d in docs:
                c1, c2 = st.columns([4, 1])
                c1.write(f"{d['source']} ({d['chunks']} chunks)")
                if c2.button("🗑", key=f"del-{d['source']}"):
                    requests.delete(f"{API}/documents/{d['source']}", timeout=10)
                    st.rerun()
        else:
            st.caption("No documents yet.")
    except requests.exceptions.ConnectionError:
        st.caption("Backend offline.")


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**[{s['id']}] {s['source']}, page {s['page']}** (score {s['score']})")
                st.caption(s["snippet"] + "...")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{API}/ask", json={"question": question}, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    answer, sources = data["answer"], data["sources"]
                else:
                    answer, sources = f"Error: {r.json().get('detail', r.text)}", []
            except requests.exceptions.ConnectionError:
                answer, sources = "Backend not running. Start it with: uvicorn app.main:app --reload", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})