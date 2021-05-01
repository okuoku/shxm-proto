#version 450 core
precision highp float;
precision highp int;











uniform webgl_input_x {
    mat4 modelMatrix;
    mat4 modelViewMatrix;
    mat4 projectionMatrix;
    mat4 viewMatrix;
    mat3 normalMatrix;
    vec3 cameraPosition;
    bool isOrthographic;
    mat3 uvTransform;
} webgl_input;

out vec4 webgl_Position;
//out float webgl_PointSize;





in vec3 position;
in vec3 normal;
in vec2 uv;































out vec3 vViewPosition;

    out vec3 vNormal;















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




        out vec2 vUv;








































































































void main(){

        vUv =(webgl_input.uvTransform * vec3(uv, 1)). xy;















    vec3 objectNormal = vec3(normal);




























    vec3 transformedNormal = objectNormal;





    transformedNormal = webgl_input.normalMatrix * transformedNormal;










        vNormal = normalize(transformedNormal);





    vec3 transformed = vec3(position);

























    vec4 mvPosition = vec4(transformed, 1.0);



    mvPosition = webgl_input.modelViewMatrix * mvPosition;
    webgl_Position = webgl_input.projectionMatrix * mvPosition;














    vViewPosition = - mvPosition . xyz;

        vec4 worldPosition = vec4(transformed, 1.0);



        worldPosition = webgl_input.modelMatrix * worldPosition;



















}

