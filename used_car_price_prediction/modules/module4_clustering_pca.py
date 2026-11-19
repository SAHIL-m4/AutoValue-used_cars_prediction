"""
Module 4: Clustering & Dimensionality Reduction (Unit 3).
Implements:
- K-Means Clustering
- Gaussian Mixture Models (GMM)
- Hierarchical (Agglomerative) Clustering & Dendrograms
- Principal Component Analysis (PCA) 2D & 3D Projections
- Elbow Method & Silhouette Score Diagnostics
- Vehicle Cluster Profiling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from utils.styling import render_section_title, render_metric_card

def render_clustering_pca_page(df: pd.DataFrame, target_col: str, inspection_report: dict):
    """Renders Unit 3 Clustering and Unsupervised Dimensionality Reduction."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #102A43 0%, #1E3E62 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #008DDA; margin-bottom: 24px;">
        <h2 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 1.6rem;">🔬 Unit 3: Clustering & Dimensionality Reduction</h2>
        <p style="color: #A0C4E2; margin: 0; font-size: 0.95rem;">
            Unsupervised pattern discovery: Segmenting vehicle market tiers with K-Means, GMM, Hierarchical clustering, and PCA manifold projections.
        </p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    
    # Feature Selection for Unsupervised Learning
    with st.expander("⚙️ Unsupervised Feature Engineering & Scaling", expanded=False):
        selected_cluster_features = st.multiselect(
            "Select Continuous Features for Clustering:",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )

    if len(selected_cluster_features) < 2:
        st.warning("Please select at least 2 numerical features for clustering and PCA.")
        return

    # Prepare and scale data
    X_raw = df[selected_cluster_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tab_kmeans, tab_gmm, tab_hier, tab_pca, tab_eval = st.tabs([
        "📍 K-Means Clustering",
        "🌌 Gaussian Mixture Models (GMM)",
        "🌳 Hierarchical Clustering",
        "📉 Principal Component Analysis (PCA)",
        "📊 Elbow & Silhouette Diagnostics"
    ])

    # ==========================================
    # TAB 1: K-Means Clustering
    # ==========================================
    with tab_kmeans:
        render_section_title("K-Means Partitioning")
        k_col1, k_col2 = st.columns([1, 1])
        with k_col1:
            k_val = st.slider("Number of Clusters (k):", 2, 8, 3, key="kmeans_k")
        with k_col2:
            init_method = st.selectbox("Centroid Initialization:", ["k-means++", "random"], key="kmeans_init")

        kmeans = KMeans(n_clusters=k_val, init=init_method, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        sil_km = silhouette_score(X_scaled, kmeans_labels)

        k1, k2, k3 = st.columns(3)
        with k1:
            render_metric_card("Clusters (k)", f"{k_val}", "Market Segments")
        with k2:
            render_metric_card("Inertia (WCSS)", f"{kmeans.inertia_:,.1f}", "Within-Cluster Variance")
        with k3:
            render_metric_card("Silhouette Score", f"{sil_km:.3f}", "Cluster Cohesion & Separation")

        # 2D Scatter using PCA
        pca_2d = PCA(n_components=2, random_state=42)
        X_pca_2d = pca_2d.fit_transform(X_scaled)

        df_viz = X_raw.copy()
        df_viz["Cluster"] = [f"Cluster {l+1}" for l in kmeans_labels]
        df_viz["PC1"] = X_pca_2d[:, 0]
        df_viz["PC2"] = X_pca_2d[:, 1]
        if target_col in df.columns:
            df_viz[target_col] = df.loc[X_raw.index, target_col]

        fig_km = px.scatter(
            df_viz, x="PC1", y="PC2", color="Cluster",
            hover_data=selected_cluster_features,
            title=f"K-Means (k={k_val}) Projected onto Principal Component Space",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_km.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
        st.plotly_chart(fig_km, use_container_width=True)

        # Cluster Profiles Table
        render_section_title("Cluster Centroid Profiles (Unscaled Means)")
        profile_df = df_viz.groupby("Cluster")[selected_cluster_features].mean()
        profile_df["Vehicle Count"] = df_viz.groupby("Cluster").size()
        st.dataframe(profile_df.style.format("{:,.2f}"), use_container_width=True)

    # ==========================================
    # TAB 2: Gaussian Mixture Models
    # ==========================================
    with tab_gmm:
        render_section_title("Probabilistic Gaussian Mixture Models (Soft Clustering)")
        st.markdown(r"""
        GMM assumes each cluster follows a multivariate normal distribution $\mathcal{N}(\mu_k, \Sigma_k)$ with prior mixture weights $\pi_k$:
        $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
        """)

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            gmm_k = st.slider("Number of Mixture Components:", 2, 8, 3, key="gmm_k")
        with g_col2:
            cov_type = st.selectbox("Covariance Type:", ["full", "tied", "diag", "spherical"], key="gmm_cov")

        gmm = GaussianMixture(n_components=gmm_k, covariance_type=cov_type, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_bic = gmm.bic(X_scaled)
        gmm_aic = gmm.aic(X_scaled)
        gmm_sil = silhouette_score(X_scaled, gmm_labels)

        gm1, gm2, gm3 = st.columns(3)
        with gm1:
            render_metric_card("GMM Silhouette", f"{gmm_sil:.3f}", "Cluster Quality")
        with gm2:
            render_metric_card("BIC Score", f"{gmm_bic:,.1f}", "Bayesian Information Criterion")
        with gm3:
            render_metric_card("AIC Score", f"{gmm_aic:,.1f}", "Akaike Information Criterion")

        df_viz["GMM_Cluster"] = [f"Mixture {l+1}" for l in gmm_labels]
        fig_gmm = px.scatter(
            df_viz, x="PC1", y="PC2", color="GMM_Cluster",
            hover_data=selected_cluster_features,
            title=f"GMM ({gmm_k} Components, {cov_type} covariance) Projection",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_gmm.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
        st.plotly_chart(fig_gmm, use_container_width=True)

    # ==========================================
    # TAB 3: Hierarchical Clustering
    # ==========================================
    with tab_hier:
        render_section_title("Hierarchical Agglomerative Clustering & Dendrogram")
        h_c1, h_c2 = st.columns(2)
        with h_c1:
            linkage_crit = st.selectbox("Linkage Method:", ["ward", "complete", "average", "single"], key="hier_link")
        with h_c2:
            hier_k = st.slider("Hierarchical Clusters Cutoff (k):", 2, 6, 3, key="hier_k")

        # Dendrogram computation on sampled records for responsive visualization
        sample_size = min(150, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        Z = linkage(X_sample, method=linkage_crit)

        fig_dendro = go.Figure()
        dendro = dendrogram(Z, no_plot=True)
        for i, d in zip(dendro['icoord'], dendro['dcoord']):
            fig_dendro.add_trace(go.Scatter(x=i, y=d, mode='lines', line=dict(color='#008DDA', width=1.5), showlegend=False))

        fig_dendro.update_layout(
            title=f"Hierarchical Dendrogram ({linkage_crit.title()} Linkage, N={sample_size} sample)",
            xaxis_title="Vehicle Sample Index", yaxis_title="Euclidean Distance Threshold",
            template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
        )
        st.plotly_chart(fig_dendro, use_container_width=True)

        # Agglomerative fit
        agg = AgglomerativeClustering(n_clusters=hier_k, linkage=linkage_crit)
        agg_labels = agg.fit_predict(X_scaled)
        agg_sil = silhouette_score(X_scaled, agg_labels)
        st.caption(f"Hierarchical Clustering with k={hier_k} achieves a **Silhouette Score of {agg_sil:.3f}**.")

    # ==========================================
    # TAB 4: PCA
    # ==========================================
    with tab_pca:
        render_section_title("Principal Component Analysis (PCA) Dimensionality Reduction")
        pca_full = PCA(random_state=42)
        pca_full.fit(X_scaled)

        exp_var = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        # Scree Plot
        scree_col1, scree_col2 = st.columns([1, 1])
        with scree_col1:
            pc_indices = [f"PC{i+1}" for i in range(len(exp_var))]
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(x=pc_indices, y=exp_var, name='Individual Variance', marker_color='#008DDA'))
            fig_scree.add_trace(go.Scatter(x=pc_indices, y=cum_var, mode='lines+markers', name='Cumulative Variance', line=dict(color='#F1F6F9', width=2.5)))
            fig_scree.update_layout(
                title="PCA Scree Plot (Explained Variance)",
                yaxis=dict(title="Proportion of Variance", tickformat=".1%"),
                template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
            )
            st.plotly_chart(fig_scree, use_container_width=True)

        with scree_col2:
            st.write("**Principal Component Loadings (Feature Contributions):**")
            loadings_df = pd.DataFrame(
                pca_full.components_[:min(3, len(pca_full.components_))].T,
                columns=[f"PC{i+1}" for i in range(min(3, len(pca_full.components_)))],
                index=selected_cluster_features
            )
            st.dataframe(loadings_df.style.format("{:+.3f}").background_gradient(cmap="Blues"), use_container_width=True)

        # 3D PCA Projection
        if len(selected_cluster_features) >= 3:
            render_section_title("Interactive 3D Principal Component Projection")
            pca_3d = PCA(n_components=3, random_state=42)
            X_pca_3d = pca_3d.fit_transform(X_scaled)
            df_viz["PC3"] = X_pca_3d[:, 2]

            fig_3d = px.scatter_3d(
                df_viz, x="PC1", y="PC2", z="PC3",
                color="Cluster",
                title="3D PCA Subspace Representation",
                template="plotly_dark",
                opacity=0.8
            )
            fig_3d.update_layout(paper_bgcolor="#0B192C")
            st.plotly_chart(fig_3d, use_container_width=True)

    # ==========================================
    # TAB 5: Elbow & Silhouette Diagnostics
    # ==========================================
    with tab_eval:
        render_section_title("Optimal Cluster Number Selection: Elbow Method & Silhouette")
        
        with st.spinner("Computing WCSS and Silhouette Scores across k = 2 ... 8:"):
            k_range = list(range(2, 9))
            wcss_list = []
            sil_list = []

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                wcss_list.append(km.inertia_)
                sil_list.append(silhouette_score(X_scaled, km.labels_))

            diag_c1, diag_c2 = st.columns(2)
            with diag_c1:
                # Elbow Plot
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=k_range, y=wcss_list, mode='lines+markers',
                    line=dict(color='#008DDA', width=3),
                    marker=dict(size=8, color='#FFFFFF')
                ))
                fig_elbow.update_layout(
                    title="The Elbow Method (Within-Cluster Sum of Squares)",
                    xaxis_title="Number of Clusters (k)", yaxis_title="Inertia / WCSS",
                    template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
                )
                st.plotly_chart(fig_elbow, use_container_width=True)

            with diag_c2:
                # Silhouette Plot
                best_k = k_range[np.argmax(sil_list)]
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=k_range, y=sil_list, mode='lines+markers',
                    line=dict(color='#41C9E2', width=3),
                    marker=dict(size=8, color='#10B981')
                ))
                fig_sil.update_layout(
                    title=f"Silhouette Analysis (Optimal Peak at k = {best_k})",
                    xaxis_title="Number of Clusters (k)", yaxis_title="Average Silhouette Coefficient",
                    template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
                )
                st.plotly_chart(fig_sil, use_container_width=True)
