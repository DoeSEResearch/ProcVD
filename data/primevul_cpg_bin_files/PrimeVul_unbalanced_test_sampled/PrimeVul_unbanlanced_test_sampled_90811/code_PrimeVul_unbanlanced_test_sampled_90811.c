  UpdateModifiedTimeTask(
      QuotaManager* manager,
      const GURL& origin,
      StorageType type,
      base::Time modified_time)
      : DatabaseTaskBase(manager),
        origin_(origin),
        type_(type),
        modified_time_(modified_time) {}
