const fs=require("fs")
const brain=require("brain.js")

const net=new brain.NeuralNetwork({
    hiddenLayers:[20,20]
})

const history=[]

const data=fs.readFileSync("history.txt","utf8")

data.split("\n").forEach(line=>{

    if(line){

        const p=line.split(",")

        history.push(p[2]=="Tài"?1:0)

    }

})


const h=history.slice(-10000)

const training=[]

for(let i=10;i<h.length;i++){

    training.push({

        input:[
            h[i-10],
            h[i-9],
            h[i-8],
            h[i-7],
            h[i-6],
            h[i-5],
            h[i-4],
            h[i-3],
            h[i-2],
            h[i-1]
        ],

        output:[h[i]]

    })

}

net.train(training,{
    iterations:3000
})

fs.writeFileSync(
    "model.json",
    JSON.stringify(net.toJSON())
)

console.log("AI trained")
