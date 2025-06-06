static Jsi_RC SqliteBackupCmd(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
                           Jsi_Value **ret, Jsi_Func *funcPtr)
{
    Jsi_Db *jdb;
    Jsi_RC rv = JSI_OK;
    int rc;
    const char *zDestFile;
    const char *zSrcDb;
    sqlite3 *pDest;
    sqlite3_backup *pBackup;

    if (!(jdb = dbGetDbHandle(interp, _this, funcPtr))) return JSI_ERROR;
    Jsi_Value *vFile = Jsi_ValueArrayIndex(interp, args, 0);
    int argc = Jsi_ValueGetLength(interp, args);
    if( argc==1 ) {
        zSrcDb = "main";
    } else {
        zSrcDb = Jsi_ValueArrayIndexToStr(interp, args, 1, NULL);
    }
    Jsi_DString dStr = {};
    if (!vFile)
        zDestFile = ":memory:";
    else {
        zDestFile = Jsi_ValueNormalPath(interp, vFile, &dStr);
        if (zDestFile == NULL) 
            return Jsi_LogError("bad or missing file name");
    }
    rc = sqlite3_open(zDestFile, &pDest);
    if( rc!=SQLITE_OK ) {
        Jsi_LogError("cannot open target database %s: %s", zDestFile, sqlite3_errmsg(pDest));
        DbClose(pDest);
        Jsi_DSFree(&dStr);
        return JSI_ERROR;
    }
    pBackup = sqlite3_backup_init(pDest, "main", jdb->db, zSrcDb);
    if( pBackup==0 ) {
        Jsi_LogError("backup failed: %s", sqlite3_errmsg(pDest));
        DbClose(pDest);
        Jsi_DSFree(&dStr);
        return JSI_ERROR;
    }
    while(  (rc = sqlite3_backup_step(pBackup,100))==SQLITE_OK ) {}
    sqlite3_backup_finish(pBackup);
    if( rc==SQLITE_DONE ) {
        rv = JSI_OK;
    } else {
        Jsi_LogError("backup failed: %s", sqlite3_errmsg(pDest));
        rv = JSI_ERROR;
    }
    Jsi_DSFree(&dStr);
    DbClose(pDest);
    return rv;
}