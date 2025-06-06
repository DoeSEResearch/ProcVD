static Jsi_RC MySqlConfCmd(Jsi_Interp *interp, Jsi_Value *args, Jsi_Value *_this,
                         Jsi_Value **ret, Jsi_Func *funcPtr)
{
    MySqlObj *jdb;
    if (!(jdb = _mysql_getDbHandle(interp, _this, funcPtr))) return JSI_ERROR;
    int oms = jdb->maxStmts;
    const char *oldu = jdb->user, *oldpw = jdb->password, *olddb = jdb->database;
    Jsi_RC rc = Jsi_OptionsConf(interp, SqlOptions, jdb, Jsi_ValueArrayIndex(interp, args, 0), ret, 0);
    if (jdb->maxStmts<0 || jdb->maxStmts>MAX_PREPARED_STMTS) {
        JSI_DBQUERY_PRINTF( "option maxStmts value %d is not in range 0..%d", jdb->maxStmts, MAX_PREPARED_STMTS);
        jdb->maxStmts = oms;
        rc = JSI_ERROR;
    }
    if (oldu != jdb->user || oldpw != jdb->password || olddb != jdb->database)
        if (!mysql_change_user(jdb->db, jdb->user, jdb->password, jdb->database)) {
            rc = JSI_ERROR;
            jdb->user = oldu;
            jdb->password = oldpw;
            jdb->database = olddb;
        }
    mdbStmtLimit(jdb);
    return rc;
}