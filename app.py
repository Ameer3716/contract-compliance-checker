import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Contract Compliance Checker", layout="wide")
st.title("Contract Compliance Checker")
st.caption("Automated Contract Analysis with TinyLlama and RAG")

@st.cache_resource
def load_system():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        if not os.path.exists("compliance_faiss_index"):
            st.error("FAISS index not found. Please ensure compliance_faiss_index folder is in the repository.")
            st.stop()
        
        vectorstore = FAISS.load_local(
            "compliance_faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        if os.path.exists("compliance_rules.json"):
            with open("compliance_rules.json", "r") as f:
                rules = json.load(f)
        else:
            st.error("compliance_rules.json not found!")
            st.stop()
        
        return vectorstore, llm, rules
    except Exception as e:
        st.error(f"Error loading system: {str(e)}")
        raise e

def check_compliance(vectorstore, llm, rule_id: str, rule_data: dict):
    rule_text = rule_data['rule']
    keywords = rule_data['keywords']
    
    query = f"Find information about {rule_text}. Look for: {', '.join(keywords)}"
    docs = vectorstore.similarity_search(query, k=3)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""<|system|>
You are a compliance checker. Analyze if the contract complies with the rule.</s>
<|user|>
Rule: {rule_text}
Contract Sections:
{context}
Does this contract comply with the rule? Answer with:
1. COMPLIANT or NON-COMPLIANT
2. Evidence from the contract
3. If non-compliant, provide remediation steps</s>
<|assistant|>
"""
    
    response = llm(prompt)
    if '<|assistant|>' in response:
        response = response.split('<|assistant|>')[-1].strip()
    
    is_compliant = "COMPLIANT" in response.upper() and "NON-COMPLIANT" not in response.upper()
    
    return {
        'rule_id': rule_id,
        'rule': rule_text,
        'category': rule_data['category'],
        'status': 'COMPLIANT' if is_compliant else 'NON-COMPLIANT',
        'evidence': docs[0].page_content[:300] if docs else "No evidence found",
        'analysis': response[:500],
        'source': docs[0].metadata.get('source', 'Unknown') if docs else 'Unknown'
    }

with st.spinner("Loading AI models and compliance rules (this may take 2-3 minutes)..."):
    try:
        vectorstore, llm, rules = load_system()
        st.success("System ready! You can now check compliance.")
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        st.stop()

st.sidebar.header("Compliance Rules")
st.sidebar.markdown(f"**Total Rules:** {len(rules)}")

categories = {}
for rule_id, rule_data in rules.items():
    cat = rule_data['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(rule_id)

st.sidebar.markdown("**Categories:**")
for cat, rule_ids in categories.items():
    st.sidebar.markdown(f"- {cat} ({len(rule_ids)})")

tab1, tab2, tab3 = st.tabs(["Single Rule Check", "Full Compliance Scan", "View Results"])

with tab1:
    st.header("Check Single Rule")
    
    rule_options = {f"{rid}: {rules[rid]['rule'][:50]}...": rid for rid in rules.keys()}
    selected_rule_display = st.selectbox("Select a rule to check:", list(rule_options.keys()))
    selected_rule_id = rule_options[selected_rule_display]
    
    selected_rule = rules[selected_rule_id]
    st.info(f"**Category:** {selected_rule['category']}\n\n**Rule:** {selected_rule['rule']}")
    st.markdown(f"**Keywords:** {', '.join(selected_rule['keywords'])}")
    
    if st.button("Check Compliance", type="primary"):
        with st.spinner("Analyzing contract..."):
            result = check_compliance(vectorstore, llm, selected_rule_id, selected_rule)
            
            if result['status'] == 'COMPLIANT':
                st.success(f"**{result['status']}**")
            else:
                st.error(f"**{result['status']}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Evidence Found")
                st.text_area("Contract Section", result['evidence'], height=200)
                st.caption(f"Source: {Path(result['source']).name if isinstance(result['source'], str) else 'Unknown'}")
            
            with col2:
                st.markdown("### AI Analysis")
                st.text_area("Analysis", result['analysis'], height=200)

with tab2:
    st.header("Full Compliance Scan")
    st.markdown("Run a comprehensive check against all 15 compliance rules.")
    
    if 'full_scan_results' not in st.session_state:
        st.session_state.full_scan_results = None
    
    if st.button("Run Full Scan", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        for i, (rule_id, rule_data) in enumerate(rules.items(), 1):
            status_text.text(f"Checking rule {i}/{len(rules)}: {rule_id}")
            result = check_compliance(vectorstore, llm, rule_id, rule_data)
            all_results.append(result)
            progress_bar.progress(i / len(rules))
        
        st.session_state.full_scan_results = all_results
        status_text.text("Scan complete!")
        
        df_results = pd.DataFrame(all_results)
        compliant = len(df_results[df_results['status'] == 'COMPLIANT'])
        non_compliant = len(df_results[df_results['status'] == 'NON-COMPLIANT'])
        
        st.markdown("---")
        st.markdown("### Compliance Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rules", len(df_results))
        col2.metric("Compliant", compliant)
        col3.metric("Non-Compliant", non_compliant)
        
        st.markdown(f"**Compliance Rate:** {compliant/len(df_results)*100:.1f}%")
        
        st.markdown("### Detailed Results")
        display_df = df_results[['rule_id', 'category', 'rule', 'status']].copy()
        display_df['rule'] = display_df['rule'].str[:60] + '...'
        
        def color_status(val):
            color = 'green' if val == 'COMPLIANT' else 'red'
            return f'background-color: {color}; color: white'
        
        st.dataframe(
            display_df.style.applymap(color_status, subset=['status']),
            use_container_width=True
        )
        
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv,
            file_name="compliance_scan_results.csv",
            mime="text/csv"
        )

with tab3:
    st.header("View Saved Results")
    
    if st.session_state.get('full_scan_results'):
        df_results = pd.DataFrame(st.session_state.full_scan_results)
        
        filter_option = st.radio("Filter by:", ["All", "Compliant Only", "Non-Compliant Only"])
        
        if filter_option == "Compliant Only":
            filtered_df = df_results[df_results['status'] == 'COMPLIANT']
        elif filter_option == "Non-Compliant Only":
            filtered_df = df_results[df_results['status'] == 'NON-COMPLIANT']
        else:
            filtered_df = df_results
        
        st.markdown(f"**Showing {len(filtered_df)} results**")
        
        for _, row in filtered_df.iterrows():
            with st.expander(f"{row['rule_id']}: {row['category']} - {row['status']}"):
                st.markdown(f"**Rule:** {row['rule']}")
                st.markdown(f"**Status:** {row['status']}")
                st.markdown("**Evidence:**")
                st.text(row['evidence'])
                st.markdown("**Analysis:**")
                st.text(row['analysis'])
                st.caption(f"Source: {Path(row['source']).name if isinstance(row['source'], str) else 'Unknown'}")
    else:
        st.info("No results yet. Run a full scan in the 'Full Compliance Scan' tab to see results here.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Tech Stack")
st.sidebar.markdown("- TinyLlama 1.1B")
st.sidebar.markdown("- FAISS Vector Store")
st.sidebar.markdown("- MiniLM Embeddings")
st.sidebar.markdown("- LangChain")
st.sidebar.markdown("- Streamlit")
