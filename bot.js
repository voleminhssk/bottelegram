const axios=require("axios")
const fs=require("fs")
const TelegramBot=require("node-telegram-bot-api")
const predict=require("./ai")
const {exec}=require("child_process")

const TOKEN="BOT_TOKEN"
const CHAT_ID="CHAT_ID"

const API="https://phanmemdudoan.fun/apisun.php"

const bot=new TelegramBot(TOKEN)

let history=[]
let lastPeriod=0
let lastPrediction=null

let wins=0
let loses=0

let today=new Date().getDate()

let counter=0



function loadHistory(){

    if(!fs.existsSync("history.txt")) return

    const data=fs.readFileSync("history.txt","utf8")

    data.split("\n").forEach(line=>{

        if(line){

            const p=line.split(",")

            history.push(p[2])

            lastPeriod=Number(p[0])

        }

    })

}

loadHistory()



function checkNewDay(){

    const now=new Date().getDate()

    if(now!=today){

        const total=wins+loses

        const winrate=
            total>0?
            (wins/total*100).toFixed(2):0

        bot.sendMessage(CHAT_ID,
`📊 Báo cáo ngày

Thắng: ${wins}
Thua: ${loses}

Winrate: ${winrate}%`)

        wins=0
        loses=0

        today=now

    }

}



async function fetchGame(){

    try{

        const res=await axios.get(API)

        const data=res.data

        if(data.period!=lastPeriod){

            lastPeriod=data.period

            const result=
                data.result.includes("T")?
                "Tài":"Xỉu"

            history.push(result)

            fs.appendFileSync(
                "history.txt",
                `${data.period},${data.total},${result}\n`
            )


            if(lastPrediction){

                if(lastPrediction==result)
                    wins++
                else
                    loses++

            }

            counter++

            if(counter>=100){

                exec("node train.js")

                counter=0

            }


            setTimeout(()=>{

                const next=predict(history)

                const msg=

`Phiên ${data.period}: ${result}

Kèo trước: ${lastPrediction || "Chưa có"}

Dự đoán tiếp: ${next.result}
AI: ${next.confidence}%`

                bot.sendMessage(CHAT_ID,msg)

                lastPrediction=next.result

            },10000)

        }

        checkNewDay()

    }catch(e){

        console.log("API lỗi")

    }

}

setInterval(fetchGame,1000)
