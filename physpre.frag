#version 450 core
#extension GL_OES_standard_derivatives : enable
#extension GL_EXT_shader_texture_lod : enable
precision highp float;
precision highp int;












uniform webgl_input_x {
    mat4 viewMatrix;
    vec3 cameraPosition;
    bool isOrthographic;
    float toneMappingExposure;
    vec3 diffuse;
    vec3 emissive;
    float roughness;
    float metalness;
    float opacity;
    float transmission;
    float reflectivity;
    float clearcoat;
    float clearcoatRoughness;
    float envMapIntensity;
    float flipEnvMap;
    int maxMipLevel;
    bool receiveShadow;
    vec3 ambientLightColor;
    vec3 lightProbe[9];
}webgl_input;

out vec4 webgl_FragColor;
//in vec4 webgl_FragCoord;
//in vec2 webgl_PointCoord;

vec3 LinearToneMapping(vec3 color){
    return webgl_input.toneMappingExposure * color;
}
vec3 ReinhardToneMapping(vec3 color){
    color *= webgl_input.toneMappingExposure;
    return clamp(color /(vec3(1.0)+ color), 0.0, 1.0);
}
vec3 OptimizedCineonToneMapping(vec3 color){
    color *= webgl_input.toneMappingExposure;
    color = max(vec3(0.0), color - 0.004);
    return pow((color *(6.2 * color + 0.5))/(color *(6.2 * color + 1.7)+ 0.06), vec3(2.2));
}
vec3 RRTAndODTFit(vec3 v){
    vec3 a = v *(v + 0.0245786)- 0.000090537;
    vec3 b = v *(0.983729 * v + 0.4329510)+ 0.238081;
    return a / b;
}
vec3 ACESFilmicToneMapping(vec3 color){
    const mat3 ACESInputMat = mat3(
    vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777)
    );
    const mat3 ACESOutputMat = mat3(
    vec3(1.60475, - 0.10208, - 0.00327), vec3(- 0.53108, 1.10813, - 0.07276), vec3(- 0.07367, - 0.00605, 1.07602)
    );
    color *= webgl_input.toneMappingExposure / 0.6;
    color = ACESInputMat * color;
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;
    return clamp(color, 0.0, 1.0);
}
vec3 CustomToneMapping(vec3 color){
    return color;
}
vec3 toneMapping(vec3 color){
    return ACESFilmicToneMapping(color);
}
vec4 LinearToLinear(in vec4 value){
    return value;
}
vec4 GammaToLinear(in vec4 value, in float gammaFactor){
    return vec4(pow(value . rgb, vec3(gammaFactor)), value . a);
}
vec4 LinearToGamma(in vec4 value, in float gammaFactor){
    return vec4(pow(value . rgb, vec3(1.0 / gammaFactor)), value . a);
}
vec4 sRGBToLinear(in vec4 value){
    return vec4(mix(pow(value . rgb * 0.9478672986 + vec3(0.0521327014), vec3(2.4)), value . rgb * 0.0773993808, vec3(lessThanEqual(value . rgb, vec3(0.04045)))), value . a);
}
vec4 LinearTosRGB(in vec4 value){
    return vec4(mix(pow(value . rgb, vec3(0.41666))* 1.055 - vec3(0.055), value . rgb * 12.92, vec3(lessThanEqual(value . rgb, vec3(0.0031308)))), value . a);
}
vec4 RGBEToLinear(in vec4 value){
    return vec4(value . rgb * exp2(value . a * 255.0 - 128.0), 1.0);
}
vec4 LinearToRGBE(in vec4 value){
    float maxComponent = max(max(value . r, value . g), value . b);
    float fExp = clamp(ceil(log2(maxComponent)), - 128.0, 127.0);
    return vec4(value . rgb / exp2(fExp),(fExp + 128.0)/ 255.0);
}
vec4 RGBMToLinear(in vec4 value, in float maxRange){
    return vec4(value . rgb * value . a * maxRange, 1.0);
}
vec4 LinearToRGBM(in vec4 value, in float maxRange){
    float maxRGB = max(value . r, max(value . g, value . b));
    float M = clamp(maxRGB / maxRange, 0.0, 1.0);
    M = ceil(M * 255.0)/ 255.0;
    return vec4(value . rgb /(M * maxRange), M);
}
vec4 RGBDToLinear(in vec4 value, in float maxRange){
    return vec4(value . rgb *((maxRange / 255.0)/ value . a), 1.0);
}
vec4 LinearToRGBD(in vec4 value, in float maxRange){
    float maxRGB = max(value . r, max(value . g, value . b));
    float D = max(maxRange / maxRGB, 1.0);
    D = clamp(floor(D)/ 255.0, 0.0, 1.0);
    return vec4(value . rgb *(D *(255.0 / maxRange)), D);
}
const mat3 cLogLuvM = mat3(0.2209, 0.3390, 0.4184, 0.1138, 0.6780, 0.7319, 0.0102, 0.1130, 0.2969);
vec4 LinearToLogLuv(in vec4 value){
    vec3 Xp_Y_XYZp = cLogLuvM * value . rgb;
    Xp_Y_XYZp = max(Xp_Y_XYZp, vec3(1e-6, 1e-6, 1e-6));
    vec4 vResult;
    vResult . xy = Xp_Y_XYZp . xy / Xp_Y_XYZp . z;
    float Le = 2.0 * log2(Xp_Y_XYZp . y)+ 127.0;
    vResult . w = fract(Le);
    vResult . z =(Le -(floor(vResult . w * 255.0))/ 255.0)/ 255.0;
    return vResult;
}
const mat3 cLogLuvInverseM = mat3(6.0014, - 2.7008, - 1.7996, - 1.3320, 3.1029, - 5.7721, 0.3008, - 1.0882, 5.6268);
vec4 LogLuvToLinear(in vec4 value){
    float Le = value . z * 255.0 + value . w;
    vec3 Xp_Y_XYZp;
    Xp_Y_XYZp . y = exp2((Le - 127.0)/ 2.0);
    Xp_Y_XYZp . z = Xp_Y_XYZp . y / value . y;
    Xp_Y_XYZp . x = value . x * Xp_Y_XYZp . z;
    vec3 vRGB = cLogLuvInverseM * Xp_Y_XYZp . rgb;
    return vec4(max(vRGB, 0.0), 1.0);
}
vec4 envMapTexelToLinear(vec4 value){
    return RGBEToLinear(value);
}
vec4 linearToOutputTexel(vec4 value){
    return LinearTosRGB(value);
}










in vec3 vViewPosition;

    in vec3 vNormal;















float pow2(const in float x){
    return x * x;
}
float pow3(const in float x){
    return x * x * x;
}
float pow4(const in float x){
    float x2 = x * x;
    return x2 * x2;
}
float average(const in vec3 color){
    return dot(color, vec3(0.3333));
}
highp float rand(const in vec2 uv){
    const highp float a = 12.9898, b = 78.233, c = 43758.5453;
    highp float dt = dot(uv . xy, vec2(a, b)), sn = mod(dt, 3.141592653589793);
    return fract(sin(sn)* c);
}

    float precisionSafeLength(vec3 v){
        return length(v);
    }









struct IncidentLight {
    vec3 color;
    vec3 direction;
    bool visible;
};
struct ReflectedLight {
    vec3 directDiffuse;
    vec3 directSpecular;
    vec3 indirectDiffuse;
    vec3 indirectSpecular;
};
struct GeometricContext {
    vec3 position;
    vec3 normal;
    vec3 viewDir;

        vec3 clearcoatNormal;

};
vec3 transformDirection(in vec3 dir, in mat4 matrix){
    return normalize((matrix * vec4(dir, 0.0)). xyz);
}
vec3 inverseTransformDirection(in vec3 dir, in mat4 matrix){
    return normalize((vec4(dir, 0.0)* matrix). xyz);
}
vec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal){
    float distance = dot(planeNormal, point - pointOnPlane);
    return - distance * planeNormal + point;
}
float sideOfPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal){
    return sign(dot(point - pointOnPlane, planeNormal));
}
vec3 linePlaneIntersect(in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal){
    return lineDirection *(dot(planeNormal, pointOnPlane - pointOnLine)/ dot(planeNormal, lineDirection))+ pointOnLine;
}
mat3 transposeMat3(const in mat3 m){
    mat3 tmp;
    tmp[0]= vec3(m[0]. x, m[1]. x, m[2]. x);
    tmp[1]= vec3(m[0]. y, m[1]. y, m[2]. y);
    tmp[2]= vec3(m[0]. z, m[1]. z, m[2]. z);
    return tmp;
}
float linearToRelativeLuminance(const in vec3 color){
    vec3 weights = vec3(0.2126, 0.7152, 0.0722);
    return dot(weights, color . rgb);
}
bool isPerspectiveMatrix(mat4 m){
    return m[2][3]== - 1.0;
}
vec2 equirectUv(in vec3 dir){
    float u = atan(dir . z, dir . x)* 0.15915494309189535 + 0.5;
    float v = asin(clamp(dir . y, - 1.0, 1.0))* 0.3183098861837907 + 0.5;
    return vec2(u, v);
}
vec3 packNormalToRGB(const in vec3 normal){
    return normalize(normal)* 0.5 + 0.5;
}
vec3 unpackRGBToNormal(const in vec3 rgb){
    return 2.0 * rgb . xyz - 1.0;
}
const float PackUpscale = 256. / 255.;
const float UnpackDownscale = 255. / 256.;
const vec3 PackFactors = vec3(256. * 256. * 256., 256. * 256., 256.);
const vec4 UnpackFactors = UnpackDownscale / vec4(PackFactors, 1.);
const float ShiftRight8 = 1. / 256.;
vec4 packDepthToRGBA(const in float v){
    vec4 r = vec4(fract(v * PackFactors), v);
    r . yzw -= r . xyz * ShiftRight8;
    return r * PackUpscale;
}
float unpackRGBAToDepth(const in vec4 v){
    return dot(v, UnpackFactors);
}
vec4 pack2HalfToRGBA(vec2 v){
    vec4 r = vec4(v . x, fract(v . x * 255.0), v . y, fract(v . y * 255.0));
    return vec4(r . x - r . y / 255.0, r . y, r . z - r . w / 255.0, r . w);
}
vec2 unpackRGBATo2Half(vec4 v){
    return vec2(v . x +(v . y / 255.0), v . z +(v . w / 255.0));
}
float viewZToOrthographicDepth(const in float viewZ, const in float near, const in float far){
    return(viewZ + near)/(near - far);
}
float orthographicDepthToViewZ(const in float linearClipZ, const in float near, const in float far){
    return linearClipZ *(near - far)- near;
}
float viewZToPerspectiveDepth(const in float viewZ, const in float near, const in float far){
    return((near + viewZ)* far)/((far - near)* viewZ);
}
float perspectiveDepthToViewZ(const in float invClipZ, const in float near, const in float far){
    return(near * far)/((far - near)* invClipZ - far);
}














    in vec2 vUv;
























vec2 integrateSpecularBRDF(const in float dotNV, const in float roughness){
    const vec4 c0 = vec4(- 1, - 0.0275, - 0.572, 0.022);
    const vec4 c1 = vec4(1, 0.0425, 1.04, - 0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r . x * r . x, exp2(- 9.28 * dotNV))* r . x + r . y;
    return vec2(- 1.04, 1.04)* a004 + r . zw;
}
float punctualLightIntensityToIrradianceFactor(const in float lightDistance, const in float cutoffDistance, const in float decayExponent){

        float distanceFalloff = 1.0 / max(pow(lightDistance, decayExponent), 0.01);
        if(cutoffDistance > 0.0){
            distanceFalloff *= pow2(clamp(1.0 - pow4(lightDistance / cutoffDistance), 0.0, 1.0));
        }
        return distanceFalloff;






}
vec3 BRDF_Diffuse_Lambert(const in vec3 diffuseColor){
    return 0.3183098861837907 * diffuseColor;
}
vec3 F_Schlick(const in vec3 specularColor, const in float dotLH){
    float fresnel = exp2((- 5.55473 * dotLH - 6.98316)* dotLH);
    return(1.0 - specularColor)* fresnel + specularColor;
}
vec3 F_Schlick_RoughnessDependent(const in vec3 F0, const in float dotNV, const in float roughness){
    float fresnel = exp2((- 5.55473 * dotNV - 6.98316)* dotNV);
    vec3 Fr = max(vec3(1.0 - roughness), F0)- F0;
    return Fr * fresnel + F0;
}
float G_GGX_Smith(const in float alpha, const in float dotNL, const in float dotNV){
    float a2 = pow2(alpha);
    float gl = dotNL + sqrt(a2 +(1.0 - a2)* pow2(dotNL));
    float gv = dotNV + sqrt(a2 +(1.0 - a2)* pow2(dotNV));
    return 1.0 /(gl * gv);
}
float G_GGX_SmithCorrelated(const in float alpha, const in float dotNL, const in float dotNV){
    float a2 = pow2(alpha);
    float gv = dotNL * sqrt(a2 +(1.0 - a2)* pow2(dotNV));
    float gl = dotNV * sqrt(a2 +(1.0 - a2)* pow2(dotNL));
    return 0.5 / max(gv + gl, 1e-6);
}
float D_GGX(const in float alpha, const in float dotNH){
    float a2 = pow2(alpha);
    float denom = pow2(dotNH)*(a2 - 1.0)+ 1.0;
    return 0.3183098861837907 * a2 / pow2(denom);
}
vec3 BRDF_Specular_GGX(const in IncidentLight incidentLight, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float roughness){
    float alpha = pow2(roughness);
    vec3 halfDir = normalize(incidentLight . direction + viewDir);
    float dotNL = clamp(dot(normal, incidentLight . direction), 0.0, 1.0);
    float dotNV = clamp(dot(normal, viewDir), 0.0, 1.0);
    float dotNH = clamp(dot(normal, halfDir), 0.0, 1.0);
    float dotLH = clamp(dot(incidentLight . direction, halfDir), 0.0, 1.0);
    vec3 F = F_Schlick(specularColor, dotLH);
    float G = G_GGX_SmithCorrelated(alpha, dotNL, dotNV);
    float D = D_GGX(alpha, dotNH);
    return F *(G * D);
}
vec2 LTC_Uv(const in vec3 N, const in vec3 V, const in float roughness){
    const float LUT_SIZE = 64.0;
    const float LUT_SCALE =(LUT_SIZE - 1.0)/ LUT_SIZE;
    const float LUT_BIAS = 0.5 / LUT_SIZE;
    float dotNV = clamp(dot(N, V), 0.0, 1.0);
    vec2 uv = vec2(roughness, sqrt(1.0 - dotNV));
    uv = uv * LUT_SCALE + LUT_BIAS;
    return uv;
}
float LTC_ClippedSphereFormFactor(const in vec3 f){
    float l = length(f);
    return max((l * l + f . z)/(l + 1.0), 0.0);
}
vec3 LTC_EdgeVectorFormFactor(const in vec3 v1, const in vec3 v2){
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 +(0.4965155 + 0.0145206 * y)* y;
    float b = 3.4175940 +(4.1616724 + y)* y;
    float v = a / b;
    float theta_sintheta =(x > 0.0)? v : 0.5 * inversesqrt(max(1.0 - x * x, 1e-7))- v;
    return cross(v1, v2)* theta_sintheta;
}
vec3 LTC_Evaluate(const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[4]){
    vec3 v1 = rectCoords[1]- rectCoords[0];
    vec3 v2 = rectCoords[3]- rectCoords[0];
    vec3 lightNormal = cross(v1, v2);
    if(dot(lightNormal, P - rectCoords[0])< 0.0)return vec3(0.0);
    vec3 T1, T2;
    T1 = normalize(V - N * dot(V, N));
    T2 = - cross(N, T1);
    mat3 mat = mInv * transposeMat3(mat3(T1, T2, N));
    vec3 coords[4];
    coords[0]= mat *(rectCoords[0]- P);
    coords[1]= mat *(rectCoords[1]- P);
    coords[2]= mat *(rectCoords[2]- P);
    coords[3]= mat *(rectCoords[3]- P);
    coords[0]= normalize(coords[0]);
    coords[1]= normalize(coords[1]);
    coords[2]= normalize(coords[2]);
    coords[3]= normalize(coords[3]);
    vec3 vectorFormFactor = vec3(0.0);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[0], coords[1]);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[1], coords[2]);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[2], coords[3]);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[3], coords[0]);
    float result = LTC_ClippedSphereFormFactor(vectorFormFactor);
    return vec3(result);
}
vec3 BRDF_Specular_GGX_Environment(const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float roughness){
    float dotNV = clamp(dot(normal, viewDir), 0.0, 1.0);
    vec2 brdf = integrateSpecularBRDF(dotNV, roughness);
    return specularColor * brdf . x + brdf . y;
}
void BRDF_Specular_Multiscattering_Environment(const in GeometricContext geometry, const in vec3 specularColor, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter){
    float dotNV = clamp(dot(geometry . normal, geometry . viewDir), 0.0, 1.0);
    vec3 F = F_Schlick_RoughnessDependent(specularColor, dotNV, roughness);
    vec2 brdf = integrateSpecularBRDF(dotNV, roughness);
    vec3 FssEss = F * brdf . x + brdf . y;
    float Ess = brdf . x + brdf . y;
    float Ems = 1.0 - Ess;
    vec3 Favg = specularColor +(1.0 - specularColor)* 0.047619;
    vec3 Fms = FssEss * Favg /(1.0 - Ems * Favg);
    singleScatter += FssEss;
    multiScatter += Fms * Ems;
}
float G_BlinnPhong_Implicit(){
    return 0.25;
}
float D_BlinnPhong(const in float shininess, const in float dotNH){
    return 0.3183098861837907 *(shininess * 0.5 + 1.0)* pow(dotNH, shininess);
}
vec3 BRDF_Specular_BlinnPhong(const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float shininess){
    vec3 halfDir = normalize(incidentLight . direction + geometry . viewDir);
    float dotNH = clamp(dot(geometry . normal, halfDir), 0.0, 1.0);
    float dotLH = clamp(dot(incidentLight . direction, halfDir), 0.0, 1.0);
    vec3 F = F_Schlick(specularColor, dotLH);
    float G = G_BlinnPhong_Implicit();
    float D = D_BlinnPhong(shininess, dotNH);
    return F *(G * D);
}
float GGXRoughnessToBlinnExponent(const in float ggxRoughness){
    return(2.0 / pow2(ggxRoughness + 0.0001)- 2.0);
}
float BlinnExponentToGGXRoughness(const in float blinnExponent){
    return sqrt(2.0 /(blinnExponent + 2.0));
}























    float getFace(vec3 direction){
        vec3 absDirection = abs(direction);
        float face = - 1.0;
        if(absDirection . x > absDirection . z){
            if(absDirection . x > absDirection . y)
            face = direction . x > 0.0 ? 0.0 : 3.0;
            else
            face = direction . y > 0.0 ? 1.0 : 4.0;
        }
        else {
            if(absDirection . z > absDirection . y)
            face = direction . z > 0.0 ? 2.0 : 5.0;
            else
            face = direction . y > 0.0 ? 1.0 : 4.0;
        }
        return face;
    }
    vec2 getUV(vec3 direction, float face){
        vec2 uv;
        if(face == 0.0){
            uv = vec2(direction . z, direction . y)/ abs(direction . x);
        }
        else if(face == 1.0){
            uv = vec2(- direction . x, - direction . z)/ abs(direction . y);
        }
        else if(face == 2.0){
            uv = vec2(- direction . x, direction . y)/ abs(direction . z);
        }
        else if(face == 3.0){
            uv = vec2(- direction . z, direction . y)/ abs(direction . x);
        }
        else if(face == 4.0){
            uv = vec2(- direction . x, direction . z)/ abs(direction . y);
        }
        else {
            uv = vec2(direction . x, direction . y)/ abs(direction . z);
        }
        return 0.5 *(uv + 1.0);
    }
    vec3 bilinearCubeUV(sampler2D envMap, vec3 direction, float mipInt){
        float face = getFace(direction);
        float filterInt = max(4.0 - mipInt, 0.0);
        mipInt = max(mipInt, 4.0);
        float faceSize = exp2(mipInt);
        float texelSize = 1.0 /(3.0 * 256.0);
        vec2 uv = getUV(direction, face)*(faceSize - 1.0);
        vec2 f = fract(uv);
        uv += 0.5 - f;
        if(face > 2.0){
            uv . y += faceSize;
            face -= 3.0;
        }
        uv . x += face * faceSize;
        if(mipInt < 8.0){
            uv . y += 2.0 * 256.0;
        }
        uv . y += filterInt * 2.0 * 16.0;
        uv . x += 3.0 * max(0.0, 256.0 - 2.0 * faceSize);
        uv *= texelSize;
        vec3 tl = envMapTexelToLinear(texture(envMap, uv)). rgb;
        uv . x += texelSize;
        vec3 tr = envMapTexelToLinear(texture(envMap, uv)). rgb;
        uv . y += texelSize;
        vec3 br = envMapTexelToLinear(texture(envMap, uv)). rgb;
        uv . x -= texelSize;
        vec3 bl = envMapTexelToLinear(texture(envMap, uv)). rgb;
        vec3 tm = mix(tl, tr, f . x);
        vec3 bm = mix(bl, br, f . x);
        return mix(tm, bm, f . y);
    }















    float roughnessToMip(float roughness){
        float mip = 0.0;
        if(roughness >= 0.8){
            mip =(1.0 - roughness)*(- 1.0 - - 2.0)/(1.0 - 0.8)+ - 2.0;
        }
        else if(roughness >= 0.4){
            mip =(0.8 - roughness)*(2.0 - - 1.0)/(0.8 - 0.4)+ - 1.0;
        }
        else if(roughness >= 0.305){
            mip =(0.4 - roughness)*(3.0 - 2.0)/(0.4 - 0.305)+ 2.0;
        }
        else if(roughness >= 0.21){
            mip =(0.305 - roughness)*(4.0 - 3.0)/(0.305 - 0.21)+ 3.0;
        }
        else {
            mip = - 2.0 * log2(1.16 * roughness);
        }
        return mip;
    }
    vec4 textureCubeUV(sampler2D envMap, vec3 sampleDir, float roughness){
        float mip = clamp(roughnessToMip(roughness), - 2.0, 8.0);
        float mipF = fract(mip);
        float mipInt = floor(mip);
        vec3 color0 = bilinearCubeUV(envMap, sampleDir, mipInt);
        if(mipF == 0.0){
            return vec4(color0, 1.0);
        }
        else {
            vec3 color1 = bilinearCubeUV(envMap, sampleDir, mipInt + 1.0);
            return vec4(mix(color0, color1, mipF), 1.0);
        }

    }





        uniform sampler2D envMap;







    vec3 getLightProbeIndirectIrradiance(const in GeometricContext geometry, const in int maxMIPLevel){
        vec3 worldNormal = inverseTransformDirection(geometry . normal, webgl_input.viewMatrix);









            vec4 envMapColor = textureCubeUV(envMap, worldNormal, 1.0);



        return 3.141592653589793 * envMapColor . rgb * webgl_input.envMapIntensity;
    }
    float getSpecularMIPLevel(const in float roughness, const in int maxMIPLevel){
        float maxMIPLevelScalar = float(maxMIPLevel);
        float sigma = 3.141592653589793 * roughness * roughness /(1.0 + roughness);
        float desiredMIPLevel = maxMIPLevelScalar + log2(sigma);
        return clamp(desiredMIPLevel, 0.0, maxMIPLevelScalar);
    }
    vec3 getLightProbeIndirectRadiance(const in vec3 viewDir, const in vec3 normal, const in float roughness, const in int maxMIPLevel){

            vec3 reflectVec = reflect(- viewDir, normal);
            reflectVec = normalize(mix(reflectVec, normal, roughness * roughness));



        reflectVec = inverseTransformDirection(reflectVec, webgl_input.viewMatrix);
        float specularMIPLevel = getSpecularMIPLevel(roughness, maxMIPLevel);









            vec4 envMapColor = textureCubeUV(envMap, reflectVec, roughness);

        return envMapColor . rgb * webgl_input.envMapIntensity;
    }











vec3 shGetIrradianceAt(in vec3 normal, in vec3 shCoefficients[9]){
    float x = normal . x, y = normal . y, z = normal . z;
    vec3 result = shCoefficients[0]* 0.886227;
    result += shCoefficients[1]* 2.0 * 0.511664 * y;
    result += shCoefficients[2]* 2.0 * 0.511664 * z;
    result += shCoefficients[3]* 2.0 * 0.511664 * x;
    result += shCoefficients[4]* 2.0 * 0.429043 * x * y;
    result += shCoefficients[5]* 2.0 * 0.429043 * y * z;
    result += shCoefficients[6]*(0.743125 * z * z - 0.247708);
    result += shCoefficients[7]* 2.0 * 0.429043 * x * z;
    result += shCoefficients[8]* 0.429043 *(x * x - y * y);
    return result;
}
vec3 getLightProbeIrradiance(const in vec3 lightProbe[9], const in GeometricContext geometry){
    vec3 worldNormal = inverseTransformDirection(geometry . normal, webgl_input.viewMatrix);
    vec3 irradiance = shGetIrradianceAt(worldNormal, lightProbe);
    return irradiance;
}
vec3 getAmbientLightIrradiance(const in vec3 ambientLightColor){
    vec3 irradiance = ambientLightColor;



    return irradiance;
}






















































































struct PhysicalMaterial {
    vec3 diffuseColor;
    float specularRoughness;
    vec3 specularColor;

        float clearcoat;
        float clearcoatRoughness;




};


float clearcoatDHRApprox(const in float roughness, const in float dotNL){
    return 0.04 +(1.0 - 0.04)*(pow(1.0 - dotNL, 5.0)* pow(1.0 - roughness, 2.0));
}


























void RE_Direct_Physical(const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight){
    float dotNL = clamp(dot(geometry . normal, directLight . direction), 0.0, 1.0);
    vec3 irradiance = dotNL * directLight . color;




        float ccDotNL = clamp(dot(geometry . clearcoatNormal, directLight . direction), 0.0, 1.0);
        vec3 ccIrradiance = ccDotNL * directLight . color;



        float clearcoatDHR = material . clearcoat * clearcoatDHRApprox(material . clearcoatRoughness, ccDotNL);
        reflectedLight . directSpecular += ccIrradiance * material . clearcoat * BRDF_Specular_GGX(directLight, geometry . viewDir, geometry . clearcoatNormal, vec3(0.04), material . clearcoatRoughness);








        reflectedLight . directSpecular +=(1.0 - clearcoatDHR)* irradiance * BRDF_Specular_GGX(directLight, geometry . viewDir, geometry . normal, material . specularColor, material . specularRoughness);

    reflectedLight . directDiffuse +=(1.0 - clearcoatDHR)* irradiance * BRDF_Diffuse_Lambert(material . diffuseColor);
}
void RE_IndirectDiffuse_Physical(const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight){
    reflectedLight . indirectDiffuse += irradiance * BRDF_Diffuse_Lambert(material . diffuseColor);
}
void RE_IndirectSpecular_Physical(const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight){

        float ccDotNV = clamp(dot(geometry . clearcoatNormal, geometry . viewDir), 0.0, 1.0);
        reflectedLight . indirectSpecular += clearcoatRadiance * material . clearcoat * BRDF_Specular_GGX_Environment(geometry . viewDir, geometry . clearcoatNormal, vec3(0.04), material . clearcoatRoughness);
        float ccDotNL = ccDotNV;
        float clearcoatDHR = material . clearcoat * clearcoatDHRApprox(material . clearcoatRoughness, ccDotNL);



    float clearcoatInv = 1.0 - clearcoatDHR;
    vec3 singleScattering = vec3(0.0);
    vec3 multiScattering = vec3(0.0);
    vec3 cosineWeightedIrradiance = irradiance * 0.3183098861837907;
    BRDF_Specular_Multiscattering_Environment(geometry, material . specularColor, material . specularRoughness, singleScattering, multiScattering);
    vec3 diffuse = material . diffuseColor *(1.0 -(singleScattering + multiScattering));
    reflectedLight . indirectSpecular += clearcoatInv * radiance * singleScattering;
    reflectedLight . indirectSpecular += multiScattering * cosineWeightedIrradiance;
    reflectedLight . indirectDiffuse += diffuse * cosineWeightedIrradiance;
}




float computeSpecularOcclusion(const in float dotNV, const in float ambientOcclusion, const in float roughness){
    return clamp(pow(dotNV + ambientOcclusion, exp2(- 16.0 * roughness - 1.0))- 1.0 + ambientOcclusion, 0.0, 1.0);
}























































































































































































































    uniform sampler2D clearcoatRoughnessMap;




















void main(){







    vec4 diffuseColor = vec4(webgl_input.diffuse, webgl_input.opacity);
    ReflectedLight reflectedLight = ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    vec3 totalEmissiveRadiance = webgl_input.emissive;

        float totalTransmission = webgl_input.transmission;




















    float roughnessFactor = webgl_input.roughness;




    float metalnessFactor = webgl_input.metalness;




    float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;





        vec3 normal = normalize(vNormal);















    vec3 geometryNormal = normal;





















        vec3 clearcoatNormal = geometryNormal;


















    PhysicalMaterial material;
    material . diffuseColor = diffuseColor . rgb *(1.0 - metalnessFactor);
    vec3 dxy = max(abs(dFdx(geometryNormal)), abs(dFdy(geometryNormal)));
    float geometryRoughness = max(max(dxy . x, dxy . y), dxy . z);
    material . specularRoughness = max(roughnessFactor, 0.0525);
    material . specularRoughness += geometryRoughness;
    material . specularRoughness = min(material . specularRoughness, 1.0);

        material . specularColor = mix(vec3(0.16 * pow2(webgl_input.reflectivity)), diffuseColor . rgb, metalnessFactor);




        material . clearcoat = webgl_input.clearcoat;
        material . clearcoatRoughness = webgl_input.clearcoatRoughness;




            material . clearcoatRoughness *= texture(clearcoatRoughnessMap, vUv). y;

        material . clearcoat = clamp(material . clearcoat, 0.0, 1.0);
        material . clearcoatRoughness = max(material . clearcoatRoughness, 0.0525);
        material . clearcoatRoughness += geometryRoughness;
        material . clearcoatRoughness = min(material . clearcoatRoughness, 1.0);





    GeometricContext geometry;
    geometry . position = - vViewPosition;
    geometry . normal = normal;
    geometry . viewDir =(webgl_input.isOrthographic)? vec3(0, 0, 1): normalize(vViewPosition);

        geometry . clearcoatNormal = clearcoatNormal;

    IncidentLight directLight;

























        vec3 iblIrradiance = vec3(0.0);
        vec3 irradiance = getAmbientLightIrradiance(webgl_input.ambientLightColor);
        irradiance += getLightProbeIrradiance(webgl_input.lightProbe, geometry);





        vec3 radiance = vec3(0.0);
        vec3 clearcoatRadiance = vec3(0.0);











            iblIrradiance += getLightProbeIndirectIrradiance(geometry, webgl_input.maxMipLevel);



        radiance += getLightProbeIndirectRadiance(geometry . viewDir, geometry . normal, material . specularRoughness, webgl_input.maxMipLevel);

            clearcoatRadiance += getLightProbeIndirectRadiance(geometry . viewDir, geometry . clearcoatNormal, material . clearcoatRoughness, webgl_input.maxMipLevel);



                         RE_IndirectDiffuse_Physical(irradiance, geometry, material, reflectedLight);


                          RE_IndirectSpecular_Physical(radiance, iblIrradiance, clearcoatRadiance, geometry, material, reflectedLight);









    vec3 outgoingLight = reflectedLight . directDiffuse + reflectedLight . indirectDiffuse + reflectedLight . directSpecular + reflectedLight . indirectSpecular + totalEmissiveRadiance;

        diffuseColor . a *= mix(clamp(1. - totalTransmission + linearToRelativeLuminance(reflectedLight . directSpecular + reflectedLight . indirectSpecular), 0.0, 1.0), 1.0, webgl_input.metalness);

    webgl_FragColor = vec4(outgoingLight, diffuseColor . a);

        webgl_FragColor . rgb = toneMapping(webgl_FragColor . rgb);

    webgl_FragColor = linearToOutputTexel(webgl_FragColor);














}

