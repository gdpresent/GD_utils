# $ pip install python-telegram-bot
import telegram
import matplotlib.pyplot as plt

def send_telegram_text(msg, bot_token, chat_id, printing=True):
    bot = telegram.Bot(token=bot_token)
    bot.send_message(chat_id, msg)
    if printing:
        print(f'텔레그램 송출 완료: {msg}')
def send_telegram_image(image_loc, bot_token, chat_id, printing=True):
    bot = telegram.Bot(token=bot_token)
    with open(image_loc, 'rb') as image_file:
        bot.send_photo(chat_id=chat_id, photo=image_file)
    if printing:
        print(f'텔레그램 송출 완료: {image_loc}')

def get_DataFrame_image(df_to_mesaage, photo_loc):
    rows, cols = df_to_mesaage.shape[0], df_to_mesaage.shape[1] - 1  # 'Group' 컬럼을 제외하기 위해 cols-1
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 0.3))  # 열과 행 수에 따라 크기 조정
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_to_mesaage.values,
                     colLabels=df_to_mesaage.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)

    # 셀 크기 조정 (행과 열 수에 따라 자동 조정)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.savefig(photo_loc, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
def get_DataFrame_image_GroupColor(df_to_mesaage, photo_loc, group_list=['date', 'SUP_GROUP']):
    df_to_mesaage['color_Group'] = df_to_mesaage[group_list].apply(tuple, axis=1)

    # 고유한 그룹 색상 매핑
    unique_groups = df_to_mesaage['color_Group'].unique()
    # 최대 9개의 명확히 구분되는 색상 리스트 설정
    distinct_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', '#FF69B4', '#8A2BE2', '#5F9EA0', '#FFA500']*10
    color_map = dict(zip(unique_groups, distinct_colors[:len(unique_groups)]))

    rows, cols = df_to_mesaage.shape[0], df_to_mesaage.shape[1] - 1  # 'Group' 컬럼을 제외하기 위해 cols-1
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 0.3))  # 열과 행 수에 따라 크기 조정
    ax.axis('tight')
    ax.axis('off')

    # table=ax.table(cellText=df_to_mesaage.values, colLabels=df_to_mesaage.columns, cellLoc='center', loc='center')
    table = ax.table(cellText=df_to_mesaage.drop(columns=['color_Group']).values,
                     colLabels=df_to_mesaage.drop(columns=['color_Group']).columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    # table.set_fontsize(font_size)

    # 행 색상 적용
    for i, group in enumerate(df_to_mesaage['color_Group']):
        for j in range(cols):
            table[(i + 1, j)].set_facecolor(color_map[group])
            table[(i + 1, j)].set_edgecolor("black")  # 셀 테두리 색상 설정
    # 셀 크기 조정 (행과 열 수에 따라 자동 조정)
    table.scale(1.2, 1.2)
    # 이미지 저장

    plt.tight_layout()
    plt.savefig(photo_loc, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
def get_Performance_image(performance_mesaage, photo_loc):
    fig, ax = plt.subplots(figsize=(10, 6))  # 그래프 크기 조정
    performance_mesaage.plot(ax=ax)
    title = f"{performance_mesaage.index[0].strftime('%Y%m%d')} - {performance_mesaage.index[-1].strftime('%Y%m%d')}"
    xlabel = "date"
    ylabel = "performance"

    # 그래프 제목과 라벨 설정
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # 여백 최소화 및 이미지 저장
    plt.tight_layout()
    plt.savefig(photo_loc, dpi=300)
    plt.close()
