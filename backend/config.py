class Config:
    UPLOAD_FOLDER = 'uploads'

    PROCESS_FLOW = {
        "select_file": "SelectFile",
        "extract_columns": "ExtractColumns",
        "edit_columns":"EditColumns",
        "create_vectordatabase":"CreateVectordatabase",
        "select_filter_parameters":"SelectFilterParameters",
        "extract_filters":"ExtractFilters",
        "merge_extracted_filters":"MergeExtractedFilters",
        "edit_filters":"EditFilters",
        "apply_filters_on_products":"ApplyFiltersOnProducts",
        "select_persona_prompt":"SelectPersonaPrompt",
        "create_persona":"CreatePersona",
        "edit_persona":"EditPersona",
        "select_user_query_prompt":"SelectUserQueryPrompt",
        "create_user_queries":"CreateUserQueries",
        "edit_user_queries":"EditUserQueries",
        "apply_filters_on_user_queries":"ApplyFiltersOnUserQueries",
        "fine_tune_model":"FineTuneModel",
        "finished":"Finished"
    }