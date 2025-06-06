  void UpdateCellularDataPlan(const CellularDataPlanList* data_plans) {
    DCHECK(cellular_);
    cellular_->SetDataPlans(data_plans);
    NotifyCellularDataPlanChanged();
  }
