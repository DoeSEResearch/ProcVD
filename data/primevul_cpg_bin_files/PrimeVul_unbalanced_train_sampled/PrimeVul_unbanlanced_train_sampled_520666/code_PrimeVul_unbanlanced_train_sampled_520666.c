static const char *mdbFindEndStr(const char *cp) {
    const char endc = *cp;
    cp++;
    while (*cp && *cp != endc) {
        if (*cp == '\\' && cp[1]) cp++;
        cp++;
    }
    if (*cp != endc)
        return NULL;
    return ++cp;
}