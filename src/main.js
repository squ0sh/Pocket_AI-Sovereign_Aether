import {CreateMLCEngine,prebuiltAppConfig} from "@mlc-ai/web-llm";
import "./style.css";
const MODELS=[
 {id:"Qwen2.5-0.5B-Instruct-q4f16_1-MLC",name:"Qwen 0.5B",tier:"Fast",description:"Working fallback · smallest option."},
 {id:"Bonsai-1.7B-q1-MLC",name:"Bonsai 1.7B Q1",tier:"Experimental",description:"Q1 model · custom MLC runtime path.",custom:true,model:"https://huggingface.co/welcoma/Bonsai-1.7B-bonsai_q1_f32-MLC/resolve/main/",model_lib:"https://huggingface.co/welcoma/Bonsai-1.7B-bonsai_q1_f32-MLC/resolve/main/libs/bonsai-q1-1.7b-bonsai_q1_f32-webgpu.wasm",overrides:{context_window_size:2048,prefill_chunk_size:128}}
];
const KEY="pocket-ai-selected-model-v2",CHATKEY="pocket-ai-chats-v4";let selected=localStorage.getItem(KEY)||MODELS[0].id,engine=null,busy=false,chats=loadChats(),active=chats[0].id;
const $=s=>document.querySelector(s),chat=$("#chat"),welcome=$("#welcome"),load=$("#loadButton"),modelButton=$("#modelButton"),sheet=$("#modelSheet"),modelList=$("#modelList"),input=$("#input"),send=$("#send"),status=$("#status"),progressWrap=$("#progressWrap"),progress=$("#progress"),progressText=$("#progressText"),errorBox=$("#errorBox"),hardwareBox=$("#hardwareBox"),drawer=$("#historyDrawer"),backdrop=$("#drawerBackdrop"),history=$("#historyList");
function model(){return MODELS.find(x=>x.id===selected)||MODELS[0]}
function renderModels(){modelList.replaceChildren();for(const m of MODELS){const b=document.createElement("button");b.className="model-option"+(m.id===selected?" active":"");const a=document.createElement("span");a.className="model-option-main";a.innerHTML=`<span class="model-option-name"></span><span class="model-option-description"></span>`;a.children[0].textContent=m.name;a.children[1].textContent=m.description;const badge=document.createElement("span");badge.className="model-option-badge";badge.textContent=m.tier;b.append(a,badge);b.onclick=()=>{selected=m.id;localStorage.setItem(KEY,selected);modelButton.textContent=m.name;sheet.classList.remove("open");if(engine) location.reload();};modelList.append(b)}}
function addBubble(role,text){const b=document.createElement("div");b.className=`message ${role}`;b.textContent=text;chat.append(b);chat.scrollTop=chat.scrollHeight;return b}
function render(){chat.querySelectorAll(".message").forEach(x=>x.remove());const c=chats.find(x=>x.id===active);if(!c||!c.messages.length){welcome.hidden=false;chat.append(welcome);return}welcome.hidden=true;c.messages.forEach(m=>addBubble(m.role,m.content))}
function loadChats(){try{const x=JSON.parse(localStorage.getItem(CHATKEY)||"null");if(Array.isArray(x)&&x.length)return x}catch{}return[{id:crypto.randomUUID(),title:"New Chat",messages:[]}]}function save(){localStorage.setItem(CHATKEY,JSON.stringify(chats));}
async function inspect(){const r={secureContext:isSecureContext,webgpu:"gpu"in navigator,adapter:false,label:"WebGPU",maxBuffer:0};if(!r.webgpu)return r;try{const a=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(a){r.adapter=true;r.maxBuffer=a.limits.maxStorageBufferBindingSize||0;const i=a.info;r.label=[i?.vendor,i?.architecture,i?.description].filter(Boolean).join(" · ")||"WebGPU"}}catch{}return r}
function showHW(h){hardwareBox.textContent=`Connection: ${h.secureContext?"HTTPS / secure":"HTTP / not secure"}\nWebGPU API: ${h.webgpu?"yes":"no"}\nGPU adapter: ${h.adapter?"yes":"no"}\nGPU: ${h.label}\nMax storage buffer: ${Math.round((h.maxBuffer||0)/1048576)} MB`}
async function loadModel(){load.disabled=true;errorBox.hidden=true;progressWrap.hidden=false;const h=await inspect();showHW(h);try{if(!h.secureContext)throw Error("WebGPU requires HTTPS.");if(!h.webgpu)throw Error("WebGPU is not available.");if(!h.adapter)throw Error("WebGPU adapter unavailable.");const m=model();
progressText.textContent=`GPU available · loading ${m.name}...`;

const appConfig={
  ...prebuiltAppConfig,
  cacheBackend:"indexeddb",
};

if(m.custom){
  appConfig.model_list=[
    ...prebuiltAppConfig.model_list,
    {
      model:m.model,
      model_id:m.id,
      model_lib:m.model_lib,
      overrides:m.overrides,
    },
  ];
}

engine=await CreateMLCEngine(m.id,{appConfig,initProgressCallback:i=>{if(i?.progress!=null)progress.style.width=Math.min(100,Math.max(0,i.progress*100))+"%";progressText.textContent=i?.text||"Preparing local GPU runtime..."}});status.textContent=`Local AI · ${m.name} · WebGPU`;progress.style.width="100%";progressText.textContent="Ready · model is running on this device.";setTimeout(()=>progressWrap.hidden=true,500);welcome.hidden=true;input.disabled=false;send.disabled=false;input.focus()}catch(e){errorBox.textContent=`WEBLLM / MLC INITIALIZATION ERROR\n\n${e.stack||e.message||e}\n\nModel: ${model().name}\nBrowser: ${navigator.userAgent}`;errorBox.hidden=false;console.error(e);load.disabled=false}}
async function sendMessage(e){e.preventDefault();const text=input.value.trim();if(!text||!engine||busy)return;busy=true;input.value="";input.disabled=true;send.disabled=true;const c=chats.find(x=>x.id===active);c.messages.push({role:"user",content:text});if(c.title==="New Chat")c.title=text.slice(0,42);save();addBubble("user",text);const a=addBubble("assistant","");try{const stream=await engine.chat.completions.create({messages:[{role:"system",content:"You are a helpful, concise assistant running locally on the user's device."},...c.messages.slice(-14)],temperature:.7,top_p:.9,max_tokens:64,stream:true});let out="";for await(const chunk of stream){out+=chunk.choices?.[0]?.delta?.content||"";a.textContent=out;chat.scrollTop=chat.scrollHeight}c.messages.push({role:"assistant",content:out.trim()});save()}catch(e){console.error("=== POCKET AI GENERATION ERROR ===");console.error("Model:",model());console.error("Error object:",e);console.error("Error message:",e?.message);console.error("Error name:",e?.name);console.error("Error stack:",e?.stack);a.textContent="GENERATION FAILED\n\n"+(e.stack||e.message||e);errorBox.textContent=`WEBLLM / MLC GENERATION ERROR\n\nModel: ${model().name}\nModel ID: ${model().id}\n\nError:\n${e.stack||e.message||e}\n\nThe model initialized successfully; this failure occurred when inference began.\n\nBrowser: ${navigator.userAgent}`;errorBox.hidden=false;status.textContent=`Local AI · ${model().name} · generation failed`;progressWrap.hidden=false;progressText.textContent="Generation failed. Model initialization succeeded.";save()}finally{busy=false;input.disabled=false;send.disabled=false;input.focus()}}
modelButton.onclick=()=>{renderModels();sheet.classList.add("open")};$("#closeModelButton").onclick=()=>sheet.classList.remove("open");$("#menuButton").onclick=()=>{drawer.classList.add("open");drawer.setAttribute("aria-hidden","false")};$("#closeDrawerButton").onclick=()=>{drawer.classList.remove("open");drawer.setAttribute("aria-hidden","true")};backdrop.onclick=()=>drawer.classList.remove("open");$("#newChatButton").onclick=()=>{const c={id:crypto.randomUUID(),title:"New Chat",messages:[]};chats.unshift(c);active=c.id;save();render();drawer.classList.remove("open")};load.onclick=loadModel;$("#composer").onsubmit=sendMessage;modelButton.textContent=model().name;renderModels();render();
if("serviceWorker"in navigator)navigator.serviceWorker.register("/sw.js").catch(console.warn);
