void QuotaManager::DidRunInitialGetTemporaryGlobalUsage(
    StorageType type, int64 usage_unused, int64 unlimited_usage_unused) {
  DCHECK_EQ(type, kStorageTypeTemporary);
  scoped_refptr<InitializeTemporaryOriginsInfoTask> task(
      new InitializeTemporaryOriginsInfoTask(
          this, temporary_usage_tracker_.get()));
  task->Start();
}
