import os
import time
import importlib

def ensure_output_dir():
    """S'assurer que le répertoire de sortie existe"""
    os.makedirs('output', exist_ok=True)
    print("Répertoire de sortie vérifié.")

def load_and_run_module(module_name):
    """Charger et exécuter un module en mesurant le temps d'exécution"""
    print(f"\n{'='*60}")
    print(f"Exécution du module: {module_name}")
    print(f"{'-'*60}")
    
    start_time = time.time()
    
    try:
        # Importer le module
        module = importlib.import_module(module_name)
        print(f"Module {module_name} importé avec succès.")
        
        # Exécuter la fonction principale du module comme si on avait exécuté le script
        # Cela équivaut à exécuter le bloc "if __name__ == '__main__'" dans chaque module
        print(f"Exécution de la fonction principale du module {module_name}...")
        if hasattr(module, 'main'):
            module.main()
        else:
            # Si le module n'a pas de fonction main(), exécuter le code comme si c'était __main__
            # Simuler l'exécution du bloc if __name__ == "__main__":
            if module_name == 'fix_country_trends':
                geo_data = module.load_geo_data()
                # Utiliser les mêmes pays que dans fix_predictions pour la cohérence
                countries_to_plot = ['United States', 'Spain', 'Singapore', 'France', 'Sweden']
                module.plot_country_trends_improved(geo_data, countries_to_plot)
            
            elif module_name == 'fix_predictions':
                geo_data = module.load_geo_data()
                countries_to_predict = ['United States', 'Spain', 'Singapore', 'France', 'Sweden']
                module.predict_country_growth_enhanced(geo_data, countries_to_predict, years_ahead=3)
            
            elif module_name == 'fix_sector_adoption':
                sector_data = module.load_sector_data()
                colors = {'primary': '#3498DB', 'secondary': '#E74C3C'}
                # Utiliser le graphique en camembert au lieu du graphique à barres
                module.plot_sector_adoption_simplified(sector_data, colors)
            
            elif module_name == 'fix_opportunity_matrix':
                sector_data = module.load_sector_data()
                growth_df = module.analyze_sector_growth(sector_data)
                colors = {
                    'admin': '#E74C3C',
                    'other': ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
                }
                module.create_opportunity_matrix_simplified(growth_df, colors)
            
            elif module_name == 'fix_gen_ai_skills':
                gen_ai_skills = module.load_gen_ai_skills_data()
                # Utiliser le graphique en camembert au lieu du graphique à barres
                module.analyze_gen_ai_skills_simplified(gen_ai_skills)
            
            elif module_name == 'fix_gen_ai_jobs':
                gen_ai_jobs = module.load_gen_ai_jobs_data()
                module.analyze_gen_ai_jobs_simplified(gen_ai_jobs)
            
            elif module_name == 'fix_top_skills':
                skills_specific = module.load_skills_data()
                module.plot_top_skills_improved(skills_specific)
            
            elif module_name == 'fix_skills_clusters':
                skills_cluster = module.load_skills_cluster_data()
                module.plot_skills_clusters_improved(skills_cluster)
            
            elif module_name == 'fix_investment_recommendations':
                geo_data, sector_data = module.load_data()
                growth_df = module.analyze_sector_growth(sector_data)
                module.generate_recommendations_simplified(geo_data, growth_df)
            
            elif module_name == 'fix_dashboard_summary':
                geo_data, sector_data, skills_specific = module.load_data()
                module.create_dashboard_summary(geo_data, sector_data, skills_specific)
    
    except ImportError as e:
        print(f"Erreur lors de l'importation du module {module_name}: {e}")
        return False
    except Exception as e:
        print(f"Erreur lors de l'exécution du module {module_name}: {e}")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Exécution terminée en {duration:.2f} secondes.")
    return True

def main():
    """Point d'entrée principal pour générer tous les graphiques"""
    print("Début de la génération de tous les graphiques...")
    
    # S'assurer que le répertoire de sortie existe
    ensure_output_dir()
    
    # Liste des modules à exécuter
    modules = [
        'fix_country_trends',          # Évolution de l'adoption par pays
        'fix_predictions',             # Prédictions de croissance
        'fix_sector_adoption',         # Adoption par secteur (graphique en camembert)
        'fix_opportunity_matrix',      # Matrice d'opportunités
        'fix_gen_ai_skills',           # Compétences en IA générative (graphique en camembert)
        'fix_gen_ai_jobs',             # Évolution des offres d'emploi en IA générative
        'fix_top_skills',              # Compétences les plus demandées
        'fix_skills_clusters',         # Évolution des compétences par groupe
        'fix_investment_recommendations', # Recommandations d'investissement
        'fix_dashboard_summary'        # Tableau de bord récapitulatif
    ]
    
    # Exécuter chaque module
    successful_runs = 0
    for module in modules:
        if load_and_run_module(module):
            successful_runs += 1
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"Génération terminée: {successful_runs}/{len(modules)} modules exécutés avec succès.")
    print(f"Les graphiques ont été enregistrés dans le dossier 'output/'.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 