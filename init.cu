#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "ComputeOptFlow.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcuof(lua_State *L);

int luaopen_libcuof(lua_State *L)
{
  lua_newtable(L);
  cuof_ComputeOptFlow_init(L);

  return 1;
}
